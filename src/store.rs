//! Where the bytes are — layer 04 §7.
//!
//! ```text
//! weights/sha256/<hex>     the ONNX file, exactly the bytes admitted
//! adapters/sha256/<hex>    the adapter document, exactly the bytes admitted
//! ```
//!
//! **The key is the hash**, so the loader derives it rather than being told a URL. Three
//! consequences, and the first is why: a replica cannot be told where to fetch from, so a
//! compromised or confused workflow cannot make it pull arbitrary bytes; `fetch_allow_hosts` is
//! empty on a replica, so only the admission instance can reach the public internet; and Kalam's
//! blob connector stays at `presign_get: false`.
//!
//! Three implementations. A directory, which is what the tests use; an HTTP base, which covers a
//! presigning proxy or a public bucket; and **S3/R2 with SigV4** — layer 07 §8.1, and a third
//! implementation of this trait rather than a change to anything above it.
//!
//! **The S3 one is what makes a fleet possible at all**, which is worth saying because it reads
//! like an optimisation and is not. The admission instance and every replica must share one store:
//! admission mirrors what it verified under the bytes' hash, and a replica fetches by that hash.
//! Two stores would mean admission succeeds and every match the version is then paired for fails
//! at the residency barrier — quietly, and a long way from the cause. Locally that sharing is one
//! volume. **Across hosts there is no shared volume**, so without this there is no fleet.

use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Weights,
    Adapter,
}

impl Kind {
    pub fn prefix(self) -> &'static str {
        match self {
            Kind::Weights => "weights",
            Kind::Adapter => "adapters",
        }
    }
}

/// `sha256:<hex>` -> `<prefix>/sha256/<hex>`. A hash that is not in that form has no key, which is
/// the first check every call makes.
pub fn key(kind: Kind, hash: &str) -> Option<String> {
    let hex = hash.strip_prefix("sha256:")?;
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(format!("{}/sha256/{}", kind.prefix(), hex))
}

pub fn digest(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("sha256:{:x}", h.finalize())
}

#[derive(Debug)]
pub enum StoreError {
    NotFound,
    Unavailable(String),
}

pub trait Store: Send + Sync {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError>;
    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError>;
    fn has(&self, kind: Kind, hash: &str) -> bool;
    fn describe(&self) -> String;
}

pub struct DirStore {
    root: PathBuf,
}

impl DirStore {
    pub fn new(root: PathBuf) -> DirStore {
        DirStore { root }
    }
    fn path(&self, kind: Kind, hash: &str) -> Option<PathBuf> {
        key(kind, hash).map(|k| self.root.join(k))
    }
}

impl Store for DirStore {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError> {
        let p = self.path(kind, hash).ok_or(StoreError::NotFound)?;
        fs::read(&p).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => StoreError::NotFound,
            _ => StoreError::Unavailable(e.to_string()),
        })
    }
    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let p = self.path(kind, hash).ok_or(StoreError::NotFound)?;
        if let Some(dir) = p.parent() {
            fs::create_dir_all(dir).map_err(|e| StoreError::Unavailable(e.to_string()))?;
        }
        // Write to a temporary name and rename, so a reader never sees a half-written object.
        // The key is the hash, so a concurrent writer is writing identical bytes and the rename
        // is idempotent by construction.
        let tmp = p.with_extension(format!("tmp{}", std::process::id()));
        fs::write(&tmp, bytes).map_err(|e| StoreError::Unavailable(e.to_string()))?;
        fs::rename(&tmp, &p).map_err(|e| StoreError::Unavailable(e.to_string()))
    }
    fn has(&self, kind: Kind, hash: &str) -> bool {
        self.path(kind, hash).map(|p| p.exists()).unwrap_or(false)
    }
    fn describe(&self) -> String {
        format!("dir:{}", self.root.display())
    }
}

pub struct HttpStore {
    base: String,
}

impl HttpStore {
    pub fn new(base: String) -> HttpStore {
        HttpStore { base }
    }
}

impl Store for HttpStore {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError> {
        let k = key(kind, hash).ok_or(StoreError::NotFound)?;
        let url = format!("{}/{k}", self.base);
        match ureq::get(&url).call() {
            Ok(mut r) => {
                let mut buf = Vec::new();
                r.body_mut()
                    .as_reader()
                    .read_to_end(&mut buf)
                    .map_err(|e| StoreError::Unavailable(e.to_string()))?;
                Ok(buf)
            }
            Err(ureq::Error::StatusCode(404)) => Err(StoreError::NotFound),
            Err(e) => Err(StoreError::Unavailable(e.to_string())),
        }
    }
    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let k = key(kind, hash).ok_or(StoreError::NotFound)?;
        let url = format!("{}/{k}", self.base);
        ureq::put(&url)
            .content_type("application/octet-stream")
            .send(bytes)
            .map(|_| ())
            .map_err(|e| StoreError::Unavailable(e.to_string()))
    }
    fn has(&self, kind: Kind, hash: &str) -> bool {
        key(kind, hash)
            .map(|k| ureq::head(&format!("{}/{k}", self.base)).call().is_ok())
            .unwrap_or(false)
    }
    fn describe(&self) -> String {
        format!("http:{}", self.base)
    }
}

// ----------------------------------------------------------------------------- S3 / R2, SigV4

/// The bytes in an S3-compatible bucket, signed with SigV4 — layer 07 §8.1.
///
/// **Path style, always**: `{endpoint}/{bucket}/{key}`. Virtual-host style would put the bucket in
/// the hostname, which R2 supports and MinIO does not without configuration, and the key here is a
/// fixed three-segment path that never needs the extra addressing.
///
/// One credential, two grants: the admission instance writes, every replica reads. Nothing in this
/// type knows which role it is in — that is `fetch_allow_hosts` and the route table, not the store.
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

    /// `host` out of the endpoint, which is what the canonical request signs. A signature over the
    /// wrong host is rejected as a signature mismatch, which reads like a bad key.
    fn host(&self) -> Result<String, StoreError> {
        let rest = self
            .endpoint
            .split_once("://")
            .map(|(_, r)| r)
            .ok_or_else(|| StoreError::Unavailable(format!("endpoint has no scheme: {}", self.endpoint)))?;
        Ok(rest.split('/').next().unwrap_or(rest).to_string())
    }

    /// Sign one request and hand back the headers to send. `payload` is the exact body; SigV4 signs
    /// its hash, so an empty body is the hash of the empty string and not a special case.
    fn sign(
        &self,
        method: &str,
        key_path: &str,
        payload: &[u8],
        now_secs: u64,
    ) -> Result<Vec<(String, String)>, StoreError> {
        let host = self.host()?;
        let (amz_date, date) = amz_timestamps(now_secs);
        let payload_hash = hex(&sha256(payload));

        // The canonical URI is the path with each segment percent-encoded and the separators kept.
        // Our segments are `<bucket>`, `weights`|`adapters`, `sha256` and 64 hex characters, none
        // of which encode to anything else -- but signing the encoded form is what the spec says,
        // and a bucket name with a dot in it would otherwise be a silent mismatch.
        let canonical_uri = format!("/{}/{}", uri_encode(&self.bucket, false), uri_encode(key_path, false));

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

    fn now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    }
}

impl Store for S3Store {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError> {
        let k = key(kind, hash).ok_or(StoreError::NotFound)?;
        let headers = self.sign("GET", &k, b"", Self::now())?;
        let mut req = ureq::get(&self.url(&k));
        for (n, v) in &headers {
            req = req.header(n, v);
        }
        match req.call() {
            Ok(mut r) => {
                let mut buf = Vec::new();
                r.body_mut()
                    .as_reader()
                    .read_to_end(&mut buf)
                    .map_err(|e| StoreError::Unavailable(e.to_string()))?;
                Ok(buf)
            }
            Err(ureq::Error::StatusCode(404)) => Err(StoreError::NotFound),
            Err(e) => Err(StoreError::Unavailable(e.to_string())),
        }
    }

    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let k = key(kind, hash).ok_or(StoreError::NotFound)?;
        let headers = self.sign("PUT", &k, bytes, Self::now())?;
        let mut req = ureq::put(&self.url(&k)).content_type("application/octet-stream");
        for (n, v) in &headers {
            req = req.header(n, v);
        }
        // The key IS the hash, so a concurrent writer is writing identical bytes and the overwrite
        // is idempotent by construction -- the same reason DirStore's rename is safe.
        req.send(bytes).map(|_| ()).map_err(|e| StoreError::Unavailable(e.to_string()))
    }

    fn has(&self, kind: Kind, hash: &str) -> bool {
        let Some(k) = key(kind, hash) else { return false };
        let Ok(headers) = self.sign("HEAD", &k, b"", Self::now()) else { return false };
        let mut req = ureq::head(&self.url(&k));
        for (n, v) in &headers {
            req = req.header(n, v);
        }
        req.call().is_ok()
    }

    fn describe(&self) -> String {
        // The access key identifies the credential and the secret never appears. `describe` is on
        // /healthz, so anything printed here is printed to whoever can reach the loader.
        format!("s3:{}/{} region={} key={}", self.endpoint, self.bucket, self.region, self.access_key)
    }
}

// ----------------------------------------------------------------------------- SigV4 primitives

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn hmac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    // `new_from_slice` rejects only a key length HMAC cannot take, and HMAC takes any length.
    let mut m = HmacSha256::new_from_slice(key).expect("HMAC accepts a key of any length");
    m.update(msg);
    m.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// AWS's unreserved set. `encode_slash` is false for a path, where `/` separates segments.
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => out.push(b as char),
            b'/' if !encode_slash => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `(20130524T000000Z, 20130524)` from a Unix timestamp.
///
/// Days-to-civil is Howard Hinnant's algorithm, shifted to an era starting 0000-03-01 so that the
/// leap day is the last day of the year and every other month has a fixed length. Chrono would do
/// this too; it is fifteen lines and one test rather than a dependency and a feature matrix.
fn amz_timestamps(secs: u64) -> (String, String) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    let date = format!("{y:04}{m:02}{d:02}");
    (format!("{date}T{h:02}{mi:02}{s:02}Z"), date)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_is_the_hash_and_nothing_else_is_a_key() {
        let h = format!("sha256:{}", "a".repeat(64));
        assert_eq!(key(Kind::Weights, &h), Some(format!("weights/sha256/{}", "a".repeat(64))));
        assert_eq!(key(Kind::Adapter, &h), Some(format!("adapters/sha256/{}", "a".repeat(64))));

        // Anything that is not a sha256 hex digest has no key. This is what stops a hash from
        // being a path: `../` cannot appear in 64 hex characters.
        assert_eq!(key(Kind::Weights, "sha256:../../etc/passwd"), None);
        assert_eq!(key(Kind::Weights, "md5:abc"), None);
        assert_eq!(key(Kind::Weights, &format!("sha256:{}", "a".repeat(63))), None);
        assert_eq!(key(Kind::Weights, &format!("sha256:{}", "g".repeat(64))), None);
        assert_eq!(key(Kind::Weights, ""), None);
    }

    #[test]
    fn the_civil_date_matches_known_timestamps() {
        // Epoch, AWS's own worked-example date, and a leap day -- the case the era shift exists
        // for, and the one an off-by-one in the algorithm gets wrong.
        assert_eq!(amz_timestamps(0), ("19700101T000000Z".into(), "19700101".into()));
        assert_eq!(amz_timestamps(1_369_353_600), ("20130524T000000Z".into(), "20130524".into()));
        assert_eq!(amz_timestamps(1_709_164_800), ("20240229T000000Z".into(), "20240229".into()));
        assert_eq!(amz_timestamps(1_709_251_199), ("20240229T235959Z".into(), "20240229".into()));
        assert_eq!(amz_timestamps(1_788_825_600), ("20260908T000000Z".into(), "20260908".into()));
    }

    #[test]
    fn uri_encoding_is_awss_unreserved_set() {
        assert_eq!(uri_encode("weights/sha256/abc", false), "weights/sha256/abc");
        assert_eq!(uri_encode("weights/sha256/abc", true), "weights%2Fsha256%2Fabc");
        assert_eq!(uri_encode("a-b_c.d~e", false), "a-b_c.d~e");
        assert_eq!(uri_encode("a b+c", false), "a%20b%2Bc");
    }

    /// Signing is the one thing here that cannot be checked by round-tripping against ourselves,
    /// because a consistently wrong implementation round-trips perfectly. So the expected values
    /// come from an INDEPENDENT implementation — Python's `hmac`/`hashlib` over the same canonical
    /// request — using AWS's publicly documented example credentials, which are safe to commit.
    ///
    /// Both verbs are pinned, because they differ in the one place SigV4 is easy to get wrong: the
    /// payload hash is part of the signature, so a GET (empty body) and a PUT (real body) exercise
    /// different halves of the canonical request.
    #[test]
    fn sigv4_matches_an_independent_implementation() {
        let s = S3Store::new(
            "https://examplebucket.s3.amazonaws.com".into(),
            "models".into(),
            "us-east-1".into(),
            "AKIAIOSFODNN7EXAMPLE".into(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
        );
        let h = format!("sha256:{}", "a".repeat(64));
        let k = key(Kind::Weights, &h).unwrap();

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
        let s = S3Store::new(
            "https://acct.r2.cloudflarestorage.com".into(),
            "models".into(),
            "auto".into(),
            "AK".into(),
            "SK".into(),
        );
        let a = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600).unwrap();
        let b = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600).unwrap();
        assert_eq!(a, b, "the same request at the same second signs identically");

        // A different day is a different signing key, which is what bounds a leaked signature.
        let c = s.sign("GET", "weights/sha256/aa", b"", 1_788_825_600 + 86_400).unwrap();
        assert_ne!(a[0].1, c[0].1);

        // A different body is a different signature: SigV4 signs the payload hash, so a swapped
        // body cannot ride an old signature.
        let d = s.sign("PUT", "weights/sha256/aa", b"one", 1_788_825_600).unwrap();
        let e = s.sign("PUT", "weights/sha256/aa", b"two", 1_788_825_600).unwrap();
        assert_ne!(d[0].1, e[0].1);
        assert_ne!(d[1].1, e[1].1);
    }

    #[test]
    fn an_s3_store_signs_the_endpoints_host_not_the_buckets() {
        let s = S3Store::new(
            "https://acct.r2.cloudflarestorage.com".into(),
            "models".into(),
            "auto".into(),
            "AK".into(),
            "SK".into(),
        );
        assert_eq!(s.host().unwrap(), "acct.r2.cloudflarestorage.com");
        assert_eq!(s.url("weights/sha256/aa"), "https://acct.r2.cloudflarestorage.com/models/weights/sha256/aa");

        // A path-style endpoint that already carries a path still signs the bare host.
        let m = S3Store::new("http://minio:9000".into(), "b".into(), "us-east-1".into(), "a".into(), "s".into());
        assert_eq!(m.host().unwrap(), "minio:9000");

        // An endpoint with no scheme is a configuration error, not a default.
        let bad = S3Store::new("minio:9000".into(), "b".into(), "r".into(), "a".into(), "s".into());
        assert!(matches!(bad.host(), Err(StoreError::Unavailable(_))));
    }

    #[test]
    fn an_s3_store_never_prints_its_secret() {
        let s = S3Store::new("https://h".into(), "b".into(), "auto".into(), "AKID".into(), "SUPERSECRET".into());
        let d = s.describe();
        assert!(d.contains("AKID"), "{d}");
        assert!(!d.contains("SUPERSECRET"), "describe() is on /healthz: {d}");
    }

    #[test]
    fn a_dir_store_round_trips() {
        let dir = std::env::temp_dir().join(format!("axon-store-{}", std::process::id()));
        let s = DirStore::new(dir.clone());
        let bytes = b"some weights".to_vec();
        let h = digest(&bytes);
        assert!(!s.has(Kind::Weights, &h));
        s.put(Kind::Weights, &h, &bytes).unwrap();
        assert!(s.has(Kind::Weights, &h));
        assert_eq!(s.get(Kind::Weights, &h).unwrap(), bytes);
        assert!(matches!(s.get(Kind::Adapter, &h), Err(StoreError::NotFound)));
        let _ = std::fs::remove_dir_all(dir);
    }
}
