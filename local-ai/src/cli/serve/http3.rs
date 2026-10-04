use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use bytes::Buf;
use http_body_util::BodyExt;

use super::{AppState, MAX_REQUEST_BYTES, route};

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
                    let Ok((request, mut stream)) = resolver.resolve_request().await else {
                        return;
                    };
                    let (parts, ()) = request.into_parts();
                    let mut data = Vec::new();
                    while let Ok(Some(mut chunk)) = stream.recv_data().await {
                        if data.len().saturating_add(chunk.remaining()) > MAX_REQUEST_BYTES {
                            return;
                        }
                        let remaining = chunk.remaining();
                        data.extend_from_slice(&chunk.copy_to_bytes(remaining));
                    }
                    let request = Request::from_parts(parts, Body::from(data));
                    let response = route(State(state), request).await;
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
