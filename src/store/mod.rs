//! Where the bytes are — docs/design.md §7.
//!
//! ```text
//! weights/sha256/<hex>     the ONNX file, exactly the bytes admitted
//! adapters/sha256/<hex>    the adapter document, exactly the bytes admitted
//! ```
//!
//! The key is the hash, so a loader derives it rather than being told a URL. Admission and every
//! replica must share one store — admission mirrors under the hash, a replica fetches by it — and
//! across hosts there is no shared volume, so [`S3Store`] is what makes a fleet possible.

mod s3;

pub use s3::S3Store;

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

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

/// `sha256:<hex>` -> `<prefix>/sha256/<hex>`. A hash not in that form has no key at all, which is
/// what stops a hash from being a path.
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

/// Drain a response body. Not `Body::read_to_vec`, which caps at 10 MiB — weights are larger.
pub(crate) fn read_body(resp: &mut ureq::http::Response<ureq::Body>) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    resp.body_mut().as_reader().read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

fn http_get(url: &str) -> Result<Vec<u8>, StoreError> {
    match ureq::get(url).call() {
        Ok(mut r) => read_body(&mut r).map_err(StoreError::Unavailable),
        Err(ureq::Error::StatusCode(404)) => Err(StoreError::NotFound),
        Err(e) => Err(StoreError::Unavailable(e.to_string())),
    }
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
        // Write then rename, so a reader never sees a half-written object. The key is the hash, so
        // a concurrent writer is writing identical bytes.
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

/// A read-mostly HTTP base: a presigning proxy or a public bucket.
pub struct HttpStore {
    base: String,
}

impl HttpStore {
    pub fn new(base: String) -> HttpStore {
        HttpStore { base }
    }

    fn url(&self, kind: Kind, hash: &str) -> Option<String> {
        key(kind, hash).map(|k| format!("{}/{k}", self.base))
    }
}

impl Store for HttpStore {
    fn get(&self, kind: Kind, hash: &str) -> Result<Vec<u8>, StoreError> {
        http_get(&self.url(kind, hash).ok_or(StoreError::NotFound)?)
    }

    fn put(&self, kind: Kind, hash: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let url = self.url(kind, hash).ok_or(StoreError::NotFound)?;
        ureq::put(&url)
            .content_type("application/octet-stream")
            .send(bytes)
            .map(|_| ())
            .map_err(|e| StoreError::Unavailable(e.to_string()))
    }

    fn has(&self, kind: Kind, hash: &str) -> bool {
        self.url(kind, hash).map(|u| ureq::head(&u).call().is_ok()).unwrap_or(false)
    }

    fn describe(&self) -> String {
        format!("http:{}", self.base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_is_the_hash_and_nothing_else_is_a_key() {
        let h = format!("sha256:{}", "a".repeat(64));
        assert_eq!(key(Kind::Weights, &h), Some(format!("weights/sha256/{}", "a".repeat(64))));
        assert_eq!(key(Kind::Adapter, &h), Some(format!("adapters/sha256/{}", "a".repeat(64))));

        // `../` cannot appear in 64 hex characters, which is what stops a hash being a path.
        assert_eq!(key(Kind::Weights, "sha256:../../etc/passwd"), None);
        assert_eq!(key(Kind::Weights, "md5:abc"), None);
        assert_eq!(key(Kind::Weights, &format!("sha256:{}", "a".repeat(63))), None);
        assert_eq!(key(Kind::Weights, &format!("sha256:{}", "g".repeat(64))), None);
        assert_eq!(key(Kind::Weights, ""), None);
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
