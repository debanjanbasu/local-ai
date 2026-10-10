use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use bytes::Buf;
use http_body_util::BodyExt;
use tokio_stream::wrappers::ReceiverStream;

use super::{AppState, route};

pub(super) async fn serve_h3(
    address: SocketAddr,
    state: AppState,
    cert_path: PathBuf,
    key_path: PathBuf,
) -> crate::Result<()> {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    let cert_data = std::fs::read(cert_path)?;
    let certs: Vec<CertificateDer<'static>> =
        rustls_pemfile::certs(&mut Cursor::new(cert_data)).collect::<Result<_, _>>()?;
    let key_data = std::fs::read(key_path)?;
    let key: PrivateKeyDer<'static> = rustls_pemfile::private_key(&mut Cursor::new(key_data))?
        .ok_or_else(|| {
            crate::Error::InvalidArgument("TLS key file contains no private key".into())
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
            let Ok(mut connection) =
                h3::server::Connection::new(h3_quinn::Connection::new(connection)).await
            else {
                return;
            };
            loop {
                let Ok(Some(resolver)) = connection.accept().await else {
                    break;
                };
                let state = state.clone();
                tokio::spawn(async move {
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
                    // The authenticated route enforces its own bounded JSON or
                    // multipart limit, identically to HTTP/1 and HTTP/2.
                    let request = Request::from_parts(
                        parts,
                        Body::from_stream(ReceiverStream::new(receiver)),
                    );
                    let response = route(State(state), request).await;
                    receiving.abort();
                    let (parts, mut body) = response.into_parts();
                    if stream
                        .send_response(axum::http::Response::from_parts(parts, ()))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    while let Some(Ok(frame)) = body.frame().await {
                        if let Ok(data) = frame.into_data()
                            && stream.send_data(data).await.is_err()
                        {
                            return;
                        }
                    }
                    let _ = stream.finish().await;
                });
            }
        });
    }
    Ok(())
}
