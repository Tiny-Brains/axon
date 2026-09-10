//! S3/R2 over SigV4 — devops/docs/deployment.md §8.1, and the only store spec that works across
//! hosts. Path style always: `{endpoint}/{bucket}/{key}`.

use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use super::{Kind, Store, StoreError, key, read_body};

type HmacSha256 = Hmac<Sha256>;
type Headers = Vec<(String, String)>;

pub struct S3Store {
    endpoint: String,
    bucket: String,
    region: String,
    access_key: String,
    secret_key: String,
}

impl S3Store {
    pub fn new(
        endpoint: String,
        bucket: String,
        region: String,
        access_key: String,
        secret_key: String,
    ) -> S3Store {
        S3Store {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            bucket,
            region,
            access_key,
            secret_key,
        }
    }

    fn url(&self, k: &str) -> String {
        format!("{}/{}/{}", self.endpoint, self.bucket, k)
    }

    /// The endpoint's host, which is what the canonical request signs. Signing the wrong host is
    /// rejected as a signature mismatch, which reads like a bad key.
    fn host(&self) -> Result<String, StoreError> {
        let rest = self.endpoint.split_once("://").map(|(_, r)| r).ok_or_else(|| {
            StoreError::Unavailable(format!("endpoint has no scheme: {}", self.endpoint))
        })?;
        Ok(rest.split('/').next().unwrap_or(rest).to_string())
    }

    /// Sign one request and hand back the headers to send. SigV4 signs the payload's hash, so an
    /// empty body is the hash of the empty string rather than a special case.
    fn sign(
        &self,
        method: &str,
        key_path: &str,
        payload: &[u8],
        now_secs: u64,
    ) -> Result<Headers, StoreError> {
        let host = self.host()?;
        let (amz_date, date) = amz_timestamps(now_secs);
        let payload_hash = hex(&sha256(payload));

        let canonical_uri = format!("/{}/{}", uri_encode(&self.bucket), uri_encode(key_path));
        let canonical_headers =
            format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
        let signed_headers = "host;x-amz-content-sha256;x-amz-date";
        let canonical_request = format!(
            "{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );

        let scope = format!("{date}/{}/s3/aws4_request", self.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            hex(&sha256(canonical_request.as_bytes()))
        );

        // The signing key is the credential walked down the scope, so a leaked signature is good
        // for one day, one region and one service.
        let k_date = hmac(format!("AWS4{}", self.secret_key).as_bytes(), date.as_bytes());
        let k_region = hmac(&k_date, self.region.as_bytes());
        let k_service = hmac(&k_region, b"s3");
        let k_signing = hmac(&k_service, b"aws4_request");
        let signature = hex(&hmac(&k_signing, string_to_sign.as_bytes()));

        Ok(vec![
            (
                "Authorization".into(),
                format!(
                    "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
                    self.access_key
                ),
            ),
            ("x-amz-content-sha256".into(), payload_hash),
            ("x-amz-date".into(), amz_date),
        ])
    }

    fn signed(
        &self,
        method: &str,
        kind: Kind,
        hash: &str,
        payload: &[u8],
    ) -> Result<(String, Headers), StoreError> {
        let k = key(kind, hash).ok_or(StoreError::NotFound)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let headers = self.sign(method, &k, payload, now)?;
        Ok((self.url(&k), headers))
    }
}

trait WithHeaders: Sized {
    fn with_headers(self, headers: &Headers) -> Self;
}

impl<S> WithHeaders for ureq::RequestBuilder<S> {
    fn with_headers(mut self, headers: &Headers) -> Self {
        for (n, v) in headers {
            self = self.header(n, v);
        }
        self
    }
}

impl Store for S3Store {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError> {
        let (url, headers) = self.signed("GET", kind, hash, b"")?;
        match ureq::get(&url).with_headers(&headers).call() {
            Ok(mut r) => read_body(&mut r).map_err(StoreError::Unavailable),
            Err(ureq::Error::StatusCode(404)) => Err(StoreError::NotFound),
            Err(e) => Err(StoreError::Unavailable(e.to_string())),
        }
    }

    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let (url, headers) = self.signed("PUT", kind, hash, bytes)?;
        // The key is the hash, so an overwrite is writing identical bytes.
        ureq::put(&url)
            .content_type("application/octet-stream")
            .with_headers(&headers)
            .send(bytes)
            .map(|_| ())
            .map_err(|e| StoreError::Unavailable(e.to_string()))
    }

    fn has(&self, kind: Kind, hash: &str) -> bool {
        match self.signed("HEAD", kind, hash, b"") {
            Ok((url, headers)) => ureq::head(&url).with_headers(&headers).call().is_ok(),
            Err(_) => false,
        }
    }

    fn describe(&self) -> String {
        // On /healthz, so the secret never appears: the access key identifies the credential.
        format!(
            "s3:{}/{} region={} key={}",
            self.endpoint, self.bucket, self.region, self.access_key
        )
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn hmac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC accepts a key of any length");
    m.update(msg);
    m.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// AWS's unreserved set, over one path whose `/` separators are kept.
fn uri_encode(s: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

/// `(20130524T000000Z, 20130524)` from a Unix timestamp. Days-to-civil is Howard Hinnant's
/// algorithm, on an era starting 0000-03-01 so the leap day falls last.
fn amz_timestamps(secs: u64) -> (String, String) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let date = format!("{y:04}{m:02}{d:02}");
    (format!("{date}T{h:02}{mi:02}{s:02}Z"), date)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::digest;

    fn store(endpoint: &str, region: &str, access: &str, secret: &str) -> S3Store {
        S3Store::new(endpoint.into(), "models".into(), region.into(), access.into(), secret.into())
    }

    fn weights_key() -> String {
        key(Kind::Weights, &format!("sha256:{}", "a".repeat(64))).unwrap()
    }

    #[test]
    fn the_civil_date_matches_known_timestamps() {
        // Epoch, AWS's worked-example date, and a leap day -- the case the era shift exists for.
        assert_eq!(amz_timestamps(0), ("19700101T000000Z".into(), "19700101".into()));
        assert_eq!(amz_timestamps(1_369_353_600), ("20130524T000000Z".into(), "20130524".into()));
        assert_eq!(amz_timestamps(1_709_164_800), ("20240229T000000Z".into(), "20240229".into()));
        assert_eq!(amz_timestamps(1_709_251_199), ("20240229T235959Z".into(), "20240229".into()));
        assert_eq!(amz_timestamps(1_788_825_600), ("20260908T000000Z".into(), "20260908".into()));
    }

    #[test]
    fn uri_encoding_is_awss_unreserved_set() {
        assert_eq!(uri_encode("weights/sha256/abc"), "weights/sha256/abc");
        assert_eq!(uri_encode("a-b_c.d~e"), "a-b_c.d~e");
        assert_eq!(uri_encode("a b+c"), "a%20b%2Bc");
    }

    /// A consistently wrong signer round-trips against itself perfectly, so the expected values
    /// come from an independent implementation — Python's `hmac`/`hashlib` over the same canonical
    /// request — using AWS's documented example credentials. Both verbs are pinned because the
    /// payload hash is part of the signature: GET and PUT exercise different halves of it.
    #[test]
    fn sigv4_matches_an_independent_implementation() {
        let s = store(
            "https://examplebucket.s3.amazonaws.com",
            "us-east-1",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        );
        let k = weights_key();

        // 2013-05-24T00:00:00Z, AWS's documented example timestamp.
        let get = s.sign("GET", &k, b"", 1_369_353_600).unwrap();
        let auth = &get.iter().find(|(n, _)| n == "Authorization").unwrap().1;
        assert!(
            auth.contains("Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request"),
            "{auth}"
        );
        assert!(auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"), "{auth}");
        assert!(
            auth.ends_with(
                "Signature=b5d9243255f708aa56c83962d0b494ebdff30cf0a682de960b376b7fb0f4b6bd"
            ),
            "{auth}"
        );

        // The empty-payload hash is a constant of the protocol and a common place to go wrong.
        let ph = &get.iter().find(|(n, _)| n == "x-amz-content-sha256").unwrap().1;
        assert_eq!(ph, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(get.iter().find(|(n, _)| n == "x-amz-date").unwrap().1, "20130524T000000Z");

        let put = s.sign("PUT", &k, b"some weights", 1_369_353_600).unwrap();
        let auth = &put.iter().find(|(n, _)| n == "Authorization").unwrap().1;
        assert!(
            auth.ends_with(
                "Signature=dc98a0bbb50fbd8fa41b4ecdb5da72eed1450d34d2d27c708d1c67a2a38cc719"
            ),
            "{auth}"
        );
        assert_eq!(
            put.iter().find(|(n, _)| n == "x-amz-content-sha256").unwrap().1,
            digest(b"some weights").strip_prefix("sha256:").unwrap()
        );
    }

    #[test]
    fn the_signing_chain_is_deterministic_and_scoped() {
        let s = store("https://acct.r2.cloudflarestorage.com", "auto", "AK", "SK");
        let a = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600).unwrap();
        let b = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600).unwrap();
        assert_eq!(a, b, "the same request at the same second signs identically");

        // A different day is a different signing key, which is what bounds a leaked signature.
        let c = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600 + 86_400).unwrap();
        assert_ne!(a[0].1, c[0].1);

        // A different body is a different signature, so a swapped body cannot ride an old one.
        let d = s.sign("PUT", "weights/sha256/aa", b"one", 1_788_825_600).unwrap();
        let e = s.sign("PUT", "weights/sha256/aa", b"two", 1_788_825_600).unwrap();
        assert_ne!(d[0].1, e[0].1);
        assert_ne!(d[1].1, e[1].1);
    }

    #[test]
    fn an_s3_store_signs_the_endpoints_host_not_the_buckets() {
        let s = store("https://acct.r2.cloudflarestorage.com", "auto", "AK", "SK");
        assert_eq!(s.host().unwrap(), "acct.r2.cloudflarestorage.com");
        assert_eq!(
            s.url("weights/sha256/aa"),
            "https://acct.r2.cloudflarestorage.com/models/weights/sha256/aa"
        );

        // A path-style endpoint that already carries a path still signs the bare host.
        assert_eq!(store("http://minio:9000", "us-east-1", "a", "s").host().unwrap(), "minio:9000");

        // An endpoint with no scheme is a configuration error, not a default.
        let bad = store("minio:9000", "r", "a", "s");
        assert!(matches!(bad.host(), Err(StoreError::Unavailable(_))));
    }

    #[test]
    fn an_s3_store_never_prints_its_secret() {
        let d = store("https://h", "auto", "AKID", "SUPERSECRET").describe();
        assert!(d.contains("AKID"), "{d}");
        assert!(!d.contains("SUPERSECRET"), "describe() is on /healthz: {d}");
    }
}
