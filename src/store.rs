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
//! Two implementations. A directory, which is what the local stack uses and what the tests use;
//! and an HTTP base, which covers a presigning proxy or a public bucket. **S3/R2 with SigV4 is
//! layer 07's**, and it is a third implementation of this trait rather than a change to anything
//! above it.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

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
    use sha2::{Digest, Sha256};
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
