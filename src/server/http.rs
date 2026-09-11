//! Routing and transport. Role checks live here: a wrong-role call is `404 NO_SUCH_CALL`.

use std::sync::Arc;

use serde::Serialize;

use super::Axon;
use crate::api::*;
use crate::config::Mode;
use crate::dialect;

type Response = tiny_http::Response<std::io::Cursor<Vec<u8>>>;

pub fn serve(axon: Arc<Axon>) -> std::io::Result<()> {
    let server = Arc::new(
        tiny_http::Server::from_listener(listen(&axon.cfg.bind)?, None)
            .map_err(|e| std::io::Error::other(e.to_string()))?,
    );
    eprintln!(
        "axon {} ({}) on {} · store {} · dialect {} · {}",
        env!("CARGO_PKG_VERSION"),
        axon.cfg.mode.name(),
        axon.cfg.bind,
        axon.store.describe(),
        dialect::DIALECT_VERSION,
        dialect::evaluator_digest(),
    );

    let mut handles = Vec::new();
    for _ in 0..axon.cfg.max_in_flight.max(1) {
        let server = server.clone();
        let axon = axon.clone();
        handles.push(std::thread::spawn(move || {
            for mut request in server.incoming_requests() {
                let response = handle(&axon, &mut request);
                let _ = request.respond(response);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// The listening socket, with `TCP_NODELAY` set so that every connection accepted from it has it
/// too — Linux and the BSDs copy the option from the listener to the accepted socket.
///
/// tiny_http writes a response's headers through a 1 KiB buffer and its body in a second write, and
/// sets no socket options. On a kept-alive connection Nagle's algorithm then holds the body until
/// the client acknowledges the headers, and the client delays that ACK — about 40 ms on Linux —
/// because it has nothing to send. Any reply over a kilobyte paid it, which is every real `/play`:
/// measured at 41 ms of a 1-match wave's ~50 ms turn on the local stack, once per turn, whatever
/// the models cost. macOS acknowledges loopback at once, so it never showed off Linux.
fn listen(bind: &str) -> std::io::Result<std::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    use std::net::ToSocketAddrs;

    let mut last = None;
    for addr in bind.to_socket_addrs()? {
        let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP))?;
        // What `TcpListener::bind` does on Unix, so a restart does not wait out TIME_WAIT.
        #[cfg(unix)]
        socket.set_reuse_address(true)?;
        socket.set_nodelay(true)?;
        match socket.bind(&addr.into()).and_then(|()| socket.listen(128)) {
            Ok(()) => return Ok(socket.into()),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| std::io::Error::other(format!("{bind} resolves to no address"))))
}

fn json<T: Serialize>(code: u16, body: &T) -> Response {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    tiny_http::Response::from_data(bytes).with_status_code(code).with_header(
        tiny_http::Header::from_bytes(&b"content-type"[..], &b"application/json"[..]).unwrap(),
    )
}

fn err(code: u16, error: &'static str, detail: Option<String>) -> Response {
    json(code, &ErrorReply { error, detail })
}

fn authorized(axon: &Axon, request: &tiny_http::Request) -> bool {
    let Some(want) = &axon.cfg.auth_token else { return true };
    let given = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("authorization"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    given.strip_prefix("Bearer ").unwrap_or("") == want
}

fn handle(axon: &Axon, request: &mut tiny_http::Request) -> Response {
    let method = request.method().as_str().to_string();
    let url = request.url().split('?').next().unwrap_or("").to_string();

    if url == "/healthz" {
        return json(
            200,
            &serde_json::json!({
                "ok": true, "mode": axon.cfg.mode.name(), "pid": std::process::id(),
                "dialect_version": dialect::DIALECT_VERSION,
                "evaluator_digest": dialect::evaluator_digest(),
                "store": axon.store.describe(),
            }),
        );
    }
    if !authorized(axon, request) {
        return err(401, "UNAUTHORIZED", None);
    }
    if method == "GET" && url == "/resident" {
        return json(200, &axon.resident());
    }
    if method != "POST" {
        return err(404, "NO_SUCH_CALL", None);
    }

    let mut body = String::new();
    if request.as_reader().read_to_string(&mut body).is_err() {
        return err(400, "MALFORMED", Some("body is not UTF-8".into()));
    }
    macro_rules! parse {
        ($t:ty) => {
            match serde_json::from_str::<$t>(&body) {
                Ok(v) => v,
                Err(e) => return err(400, "MALFORMED", Some(e.to_string())),
            }
        };
    }

    match url.as_str() {
        "/load" => json(200, &axon.load(parse!(LoadRequest))),
        "/unload" => json(200, &axon.unload(parse!(UnloadRequest))),
        "/play" if axon.cfg.mode == Mode::Admission => err(404, "NO_SUCH_CALL", None),
        "/play" => json(200, &axon.play(parse!(PlayRequest))),
        "/inspect" | "/validate" if axon.cfg.mode == Mode::Replica => {
            err(404, "NO_SUCH_CALL", None)
        }
        "/inspect" => match axon.inspect(parse!(InspectRequest)) {
            Ok(r) => json(200, &r),
            Err(e) => json(404, &e),
        },
        "/validate" => json(200, &axon.validate(parse!(ValidateRequest))),
        _ => err(404, "NO_SUCH_CALL", None),
    }
}
