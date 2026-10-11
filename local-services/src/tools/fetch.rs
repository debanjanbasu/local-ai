//! Web fetch: one `GET` to an allowlisted public host.
//!
//! - The URL must be `http` or `https`, without user information, naming a
//!   host by DNS name (IP literals are refused) that equals an allowlisted
//!   domain or is a subdomain of one.
//! - Every address the name resolves to must be public: loopback, private,
//!   link-local, shared, documentation, multicast and other special-purpose
//!   ranges (and IPv4-mapped forms of them) are refused. The connection goes
//!   to an address that was checked, so the name is never resolved again in
//!   between (no DNS rebinding).
//! - Redirects are reported, never followed: the model may fetch the target
//!   itself, through the same checks.
//! - `https` is verified by the platform verifier (the macOS trust store).
//! - The body is read up to the byte limit and the connection then dropped,
//!   with a marker saying the text was cut; the caller's deadline bounds the
//!   whole exchange and dropping the future closes the connection.

use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use http_body_util::BodyExt as _;
use hyper_util::rt::TokioIo;
use rustls::pki_types::ServerName;

use super::{Output, ToolError};

/// Characters of a URL accepted.
const MAX_URL_BYTES: usize = 4096;

/// Check one allowlist entry: a lowercase DNS name, not an IP address.
pub fn check_domain(domain: &str) -> Result<(), ToolError> {
    let valid = !domain.is_empty()
        && domain.len() <= 253
        && domain.parse::<IpAddr>().is_err()
        && domain.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        });
    if valid {
        Ok(())
    } else {
        Err(ToolError::Invalid(format!(
            "fetch domain {domain:?} must be a lowercase DNS name such as example.com"
        )))
    }
}

/// Whether `host` is `domain` or one of its subdomains, for some entry.
#[must_use]
pub fn allowed(host: &str, domains: &[String]) -> bool {
    domains.iter().any(|domain| {
        host == domain
            || host
                .strip_suffix(domain.as_str())
                .is_some_and(|prefix| prefix.ends_with('.'))
    })
}

/// Whether `ip` is an ordinary public unicast address.
#[must_use]
pub fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => ip.to_ipv4_mapped().map_or_else(|| public_v6(ip), public_v4),
    }
}

fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.is_multicast()
        || a == 0
        || (a == 100 && (64..128).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 198 && (b == 18 || b == 19))
        || a >= 240)
}

const fn public_v6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    let first = segments[0];
    // Global unicast is 2000::/3; within it, refuse documentation
    // (2001:db8::/32), 6to4 (2002::/16), Teredo and other 2001::/23
    // special-purpose blocks, which embed or reach other address spaces.
    (first & 0xe000) == 0x2000
        && first != 0x2002
        && !(first == 0x2001 && segments[1] < 0x0200)
        && !(first == 0x2001 && segments[1] == 0x0db8)
}

/// The parts of a URL a fetch needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub https: bool,
    pub host: String,
    pub port: u16,
    pub path: String,
}

/// Parse and check `url` against `domains`, before any network use.
pub fn target(url: &str, domains: &[String]) -> Result<Target, ToolError> {
    if url.len() > MAX_URL_BYTES {
        return Err(ToolError::Invalid(format!(
            "the URL is longer than {MAX_URL_BYTES} bytes"
        )));
    }
    let uri: http::Uri = url
        .parse()
        .map_err(|error| ToolError::Invalid(format!("invalid URL: {error}")))?;
    let https = match uri.scheme_str() {
        Some("https") => true,
        Some("http") => false,
        _ => {
            return Err(ToolError::Denied(
                "only http and https URLs may be fetched".into(),
            ));
        }
    };
    let authority = uri
        .authority()
        .ok_or_else(|| ToolError::Invalid("the URL has no host".into()))?;
    if authority.as_str().contains('@') {
        return Err(ToolError::Denied(
            "URLs with user information are refused".into(),
        ));
    }
    let host = authority.host().to_ascii_lowercase();
    let host = host.trim_end_matches('.').to_owned();
    if host.starts_with('[') || host.parse::<IpAddr>().is_ok() {
        return Err(ToolError::Denied(
            "IP address hosts are refused; the allowlist names domains".into(),
        ));
    }
    if !allowed(&host, domains) {
        return Err(ToolError::Denied(format!(
            "{host} is not on this server's fetch allowlist"
        )));
    }
    let port = authority.port_u16().unwrap_or(if https { 443 } else { 80 });
    let path = uri
        .path_and_query()
        .map_or_else(|| "/".to_owned(), |path| path.as_str().to_owned());
    Ok(Target {
        https,
        host,
        port,
        path,
    })
}

/// Resolve `target` and return an address, refusing the name if any of its
/// addresses is not public.
async fn resolve(target: &Target) -> Result<SocketAddr, ToolError> {
    let addresses: Vec<SocketAddr> = tokio::net::lookup_host((target.host.as_str(), target.port))
        .await
        .map_err(|error| ToolError::Failed(format!("cannot resolve {}: {error}", target.host)))?
        .collect();
    if let Some(private) = addresses.iter().find(|address| !is_public(address.ip())) {
        return Err(ToolError::Denied(format!(
            "{} resolves to the non-public address {}, which is never contacted",
            target.host,
            private.ip()
        )));
    }
    addresses
        .first()
        .copied()
        .ok_or_else(|| ToolError::Failed(format!("{} has no addresses", target.host)))
}

/// Aborts the connection task when dropped, so a cancelled fetch closes its
/// socket.
struct Connection(tokio::task::JoinHandle<()>);

impl Drop for Connection {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Fetch `url`, reading at most `max_bytes` of body.
pub async fn fetch(url: &str, domains: &[String], max_bytes: usize) -> Result<Output, ToolError> {
    let target = target(url, domains)?;
    let address = resolve(&target).await?;
    let failed =
        |error: &dyn std::fmt::Display| ToolError::Failed(format!("fetch failed: {error}"));
    let tcp = tokio::net::TcpStream::connect(address)
        .await
        .map_err(|error| failed(&error))?;
    let request = http::Request::get(target.path.as_str())
        .header(http::header::HOST, host_header(&target))
        .header(http::header::USER_AGENT, "local-ai-tools/1")
        .header(http::header::ACCEPT, "*/*")
        .header(http::header::ACCEPT_ENCODING, "identity")
        .header(http::header::CONNECTION, "close")
        .body(http_body_util::Empty::<bytes::Bytes>::new())
        .map_err(|error| failed(&error))?;
    let response = if target.https {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = rustls_platform_verifier::Verifier::new(Arc::clone(&provider))
            .map_err(|error| failed(&error))?;
        let config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| failed(&error))?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(verifier))
            .with_no_client_auth();
        let name = ServerName::try_from(target.host.clone()).map_err(|error| failed(&error))?;
        let tls = tokio_rustls::TlsConnector::from(Arc::new(config))
            .connect(name, tcp)
            .await
            .map_err(|error| failed(&error))?;
        send(TokioIo::new(tls), request).await?
    } else {
        send(TokioIo::new(tcp), request).await?
    };
    let (response, _connection) = response;
    let status = response.status();
    let mut text = format!("HTTP {}\n", status.as_u16());
    for name in [http::header::CONTENT_TYPE, http::header::LOCATION] {
        if let Some(value) = response
            .headers()
            .get(&name)
            .and_then(|value| value.to_str().ok())
        {
            let _ = writeln!(text, "{name}: {value}");
        }
    }
    if status.is_redirection() {
        text.push_str("[redirect not followed; fetch the location to continue]\n");
        return Ok(Output::Text {
            text,
            truncated: false,
        });
    }
    text.push('\n');
    let mut body = response.into_body();
    let mut kept = Vec::new();
    let mut truncated = false;
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|error| failed(&error))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        let room = max_bytes.saturating_sub(kept.len());
        if data.len() > room {
            kept.extend_from_slice(data.get(..room).unwrap_or_default());
            truncated = true;
            break;
        }
        kept.extend_from_slice(&data);
    }
    text.push_str(&String::from_utf8_lossy(&kept));
    if truncated {
        let _ = write!(text, "\n[truncated: the body was cut at {max_bytes} bytes]");
    }
    Ok(Output::Text { text, truncated })
}

fn host_header(target: &Target) -> String {
    let default = if target.https { 443 } else { 80 };
    if target.port == default {
        target.host.clone()
    } else {
        format!("{}:{}", target.host, target.port)
    }
}

async fn send<T>(
    io: T,
    request: http::Request<http_body_util::Empty<bytes::Bytes>>,
) -> Result<(http::Response<hyper::body::Incoming>, Connection), ToolError>
where
    T: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
{
    let (mut sender, connection) = hyper::client::conn::http1::handshake(io)
        .await
        .map_err(|error| ToolError::Failed(format!("fetch failed: {error}")))?;
    let connection = Connection(tokio::spawn(async move {
        let _ = connection.await;
    }));
    let response = sender
        .send_request(request)
        .await
        .map_err(|error| ToolError::Failed(format!("fetch failed: {error}")))?;
    Ok((response, connection))
}
