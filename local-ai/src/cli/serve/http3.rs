use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use bytes::{Buf, Bytes};
use http_body_util::BodyExt;
use tokio_stream::wrappers::ReceiverStream;

use super::{AppState, route};

pub(super) async fn serve_h3(
    address: SocketAddr,
    state: AppState,
    cert_path: PathBuf,
    key_path: PathBuf,
) -> crate::Result<()> {
    use rustls::pki_types::pem::{self, PemObject};
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    let cert_data = std::fs::read(cert_path)?;
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&cert_data)
        .collect::<Result<_, _>>()
        .map_err(|error| {
            crate::Error::InvalidArgument(format!("invalid TLS certificate file: {error}"))
        })?;
    let key_data = std::fs::read(key_path)?;
    let key = PrivateKeyDer::from_pem_slice(&key_data).map_err(|error| match error {
        pem::Error::NoItemsFound => {
            crate::Error::InvalidArgument("TLS key file contains no private key".into())
        }
        error => crate::Error::InvalidArgument(format!("invalid TLS key file: {error}")),
    })?;
    let mut tls = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|error| {
            crate::Error::InvalidArgument(format!("invalid TLS configuration: {error}"))
        })?;
    tls.alpn_protocols = vec![b"h3".to_vec()];
    let quic = quinn::crypto::rustls::QuicServerConfig::try_from(tls).map_err(|error| {
        crate::Error::InvalidArgument(format!("invalid QUIC TLS configuration: {error}"))
    })?;
    let endpoint =
        quinn::Endpoint::server(quinn::ServerConfig::with_crypto(Arc::new(quic)), address)?;
    eprintln!("Bonsai HTTP/3 server listening on https://{address}");
    while let Some(incoming) = endpoint.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            let Ok(connection) = incoming.await else {
                return;
            };
            // Kept to notice the client going away while a request is still
            // being answered, when nothing is being written to it.
            let quic = connection.clone();
            // The header block a request may carry, like the TCP listener's
            // bounded header buffer; h3's default is unlimited.
            let Ok(mut connection) = h3::server::builder()
                .max_field_section_size(MAX_FIELD_SECTION_BYTES)
                .build(h3_quinn::Connection::new(connection))
                .await
            else {
                return;
            };
            loop {
                let Ok(Some(resolver)) = connection.accept().await else {
                    break;
                };
                let state = state.clone();
                let quic = quic.clone();
                tokio::spawn(answer(resolver, state, quic));
            }
        });
    }
    Ok(())
}

/// Answer one HTTP/3 request on `quic`.
async fn answer(
    resolver: h3::server::RequestResolver<h3_quinn::Connection, Bytes>,
    state: AppState,
    quic: quinn::Connection,
) {
    let Ok((request, stream)) = resolver.resolve_request().await else {
        return;
    };
    let (parts, ()) = request.into_parts();
    let (mut stream, mut input) = stream.split();
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    let receiving = tokio::spawn(async move {
        loop {
            let data = match input.recv_data().await {
                Ok(Some(mut chunk)) => {
                    let size = chunk.remaining();
                    Ok(chunk.copy_to_bytes(size))
                }
                Ok(None) => break,
                Err(error) => Err(std::io::Error::other(error.to_string())),
            };
            let failed = data.is_err();
            if sender.send(data).await.is_err() || failed {
                input.stop_sending(h3::error::Code::H3_REQUEST_CANCELLED);
                break;
            }
        }
    });
    // The authenticated route enforces its own bounded JSON or multipart
    // limit, identically to HTTP/1 and HTTP/2.
    let request = Request::from_parts(parts, Body::from_stream(ReceiverStream::new(receiver)));
    // A vanished client drops the request: its handler, and with it any
    // generation it started, is cancelled.
    let response = tokio::select! {
        response = route(State(state), request) => response,
        _ = quic.closed() => {
            receiving.abort();
            return;
        }
    };
    receiving.abort();
    let (parts, mut body) = response.into_parts();
    if stream
        .send_response(axum::http::Response::from_parts(parts, ()))
        .await
        .is_err()
    {
        return;
    }
    let ended = tokio::select! {
        ended = forward(&mut body, &mut stream) => ended,
        _ = quic.closed() => Ended::Gone,
    };
    match ended {
        Ended::Complete => {
            let _ = stream.finish().await;
        }
        // A body that failed partway (a speech stream whose synthesis
        // failed) is reset, never finished, so the client cannot take the
        // truncated body for whole.
        Ended::Failed => stream.stop_stream(h3::error::Code::H3_INTERNAL_ERROR),
        Ended::Gone => {}
    }
}

/// Largest request header block accepted over HTTP/3: 64 KiB.
const MAX_FIELD_SECTION_BYTES: u64 = 64 << 10;

/// How a response body ended on its stream.
#[derive(Debug, PartialEq, Eq)]
enum Ended {
    /// Every frame was sent.
    Complete,
    /// The body failed partway.
    Failed,
    /// The client is gone.
    Gone,
}

/// Where response data goes: the HTTP/3 request stream.
trait Sink {
    /// Send `data`, `false` once the client is gone.
    fn send(&mut self, data: Bytes) -> impl Future<Output = bool> + Send;
}

impl Sink for h3::server::RequestStream<h3_quinn::SendStream<Bytes>, Bytes> {
    async fn send(&mut self, data: Bytes) -> bool {
        self.send_data(data).await.is_ok()
    }
}

/// Send `body`'s data to `sink` until it ends, fails or the client leaves.
async fn forward(body: &mut Body, sink: &mut impl Sink) -> Ended {
    loop {
        match body.frame().await {
            None => return Ended::Complete,
            Some(Err(_)) => return Ended::Failed,
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data()
                    && !sink.send(data).await
                {
                    return Ended::Gone;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use super::{Body, Bytes, Ended, Sink, forward};

    impl Sink for Vec<Bytes> {
        fn send(&mut self, data: Bytes) -> impl Future<Output = bool> + Send {
            self.push(data);
            std::future::ready(true)
        }
    }

    #[tokio::test]
    async fn a_body_that_fails_partway_is_not_reported_complete() {
        let frames = [
            Ok(Bytes::from_static(b"RIFF")),
            Err(std::io::Error::other("synthesis failed")),
        ];
        let mut failing = Body::from_stream(tokio_stream::iter(frames));
        let mut sent = Vec::new();
        assert_eq!(forward(&mut failing, &mut sent).await, Ended::Failed);
        assert_eq!(sent, [Bytes::from_static(b"RIFF")]);
        let mut whole = Body::from("RIFF....");
        let mut sent = Vec::new();
        assert_eq!(forward(&mut whole, &mut sent).await, Ended::Complete);
        assert_eq!(sent, [Bytes::from_static(b"RIFF....")]);
    }
}
