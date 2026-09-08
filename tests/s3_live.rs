//! S3Store against a real S3 implementation — devops/docs/deployment.md §8.1.
//!
//! Signing is checked against an independent implementation in the unit tests; what those cannot
//! check is whether a real server ACCEPTS the signature, which is a different question and the one
//! that matters. So this drives the local stack's MinIO, which is the same S3 surface R2 presents
//! and the same one `devops/loader/run.sh` already signs against with curl.
//!
//! **Skipped, loudly, when the stack is not up.** A test that silently passes when its subject is
//! absent is worse than no test: it reports green for a store nobody exercised.
//!
//!   docker compose up -d minio     # from devops/
//!   AXON_S3_LIVE=1 cargo test --test s3_live -- --nocapture

use axon::store::{digest, Kind, S3Store, Store, StoreError};

fn store() -> Option<S3Store> {
    if std::env::var("AXON_S3_LIVE").is_err() {
        eprintln!("SKIPPED: set AXON_S3_LIVE=1 with the compose stack up to run this");
        return None;
    }
    Some(S3Store::new(
        std::env::var("AXON_STORE_S3_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:9000".into()),
        std::env::var("AXON_STORE_S3_BUCKET").unwrap_or_else(|_| "tinybrains-replays".into()),
        std::env::var("AXON_STORE_S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
        std::env::var("AXON_STORE_S3_ACCESS_KEY").unwrap_or_else(|_| "tinybrains".into()),
        std::env::var("AXON_STORE_S3_SECRET_KEY")
            .unwrap_or_else(|_| "tinybrains-dev-secret".into()),
    ))
}

#[test]
fn a_real_s3_accepts_our_signature_and_round_trips() {
    let Some(s) = store() else { return };

    // Distinct bytes per run, so a stale object from a previous run cannot make this pass.
    let bytes = format!("axon s3 round trip {}", std::process::id()).into_bytes();
    let h = digest(&bytes);

    assert!(!s.has(Kind::Weights, &h), "a fresh hash is not in the bucket");
    s.put(Kind::Weights, &h, &bytes).expect("PUT signed and accepted");
    assert!(s.has(Kind::Weights, &h), "HEAD signed and accepted");
    assert_eq!(s.get(Kind::Weights, &h).expect("GET signed and accepted"), bytes);

    // The kind is part of the key, so the same hash under the other prefix is a different object.
    assert!(matches!(s.get(Kind::Adapter, &h), Err(StoreError::NotFound)));
}

#[test]
fn a_missing_object_is_not_found_rather_than_unavailable() {
    let Some(s) = store() else { return };
    // The distinction is load-bearing: NotFound is a competitor's asset never mirrored, and
    // Unavailable is the store being down. Admission rejects on one and retries on the other.
    let h = format!("sha256:{}", "b".repeat(64));
    assert!(matches!(s.get(Kind::Weights, &h), Err(StoreError::NotFound)));
}

#[test]
fn a_bad_secret_is_unavailable_not_not_found() {
    if std::env::var("AXON_S3_LIVE").is_err() {
        return;
    }
    let s = S3Store::new(
        std::env::var("AXON_STORE_S3_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:9000".into()),
        "tinybrains-replays".into(),
        "us-east-1".into(),
        "tinybrains".into(),
        "the-wrong-secret".into(),
    );
    let h = format!("sha256:{}", "c".repeat(64));
    // A signature mismatch is a 403, which must not be mistaken for "the competitor never
    // uploaded this" -- that would reject a good submission for a credential problem.
    match s.get(Kind::Weights, &h) {
        Err(StoreError::Unavailable(_)) => {}
        other => panic!("expected Unavailable for a bad signature, got {other:?}"),
    }
}
