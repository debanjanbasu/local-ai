//! Web fetch: one `GET` to an allowlisted public host ([`fetch`]), and the
//! same checked transport for media a request names by URL ([`download`]).
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
//!   itself, through the same checks. ([`download`] follows up to five, each
//!   target checked anew, never from `https` to `http`.)
//! - `https` is verified by the platform verifier (the macOS trust store).
//! - The transport ([`client`]) prefers HTTP/3 where an origin advertises
//!   it, falls back to HTTP/2 or HTTP/1.1 over TCP, and reuses pooled
//!   connections; any `Alt-Svc` or DNS HTTPS alternative is only used on
//!   the same host, at its checked addresses.
//! - The body arrives in any of `zstd`, `br`, `gzip` and `deflate` and is
//!   decoded while it streams ([`body`]), up to the byte limit (both decoded
//!   and encoded bytes are bounded, so a decompression bomb stops at the
//!   limit). HTML is converted to Markdown ([`markdown`]). A cut body says
//!   so with a marker; the caller's deadline bounds the whole exchange,
//!   conversion included, and dropping the future abandons the connection.

mod altsvc;
mod body;
mod client;
mod dns;
mod markdown;
#[cfg(test)]
mod network_tests;

use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

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

/// Parse `url` (scheme, user information, host and port) before any
/// network use. An IP literal host is kept, without IPv6 brackets.
fn parse(url: &str) -> Result<Target, ToolError> {
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
    let host = host
        .trim_end_matches('.')
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_owned();
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

/// Parse and check `url` against `domains`, before any network use.
pub fn target(url: &str, domains: &[String]) -> Result<Target, ToolError> {
    let target = parse(url)?;
    if target.host.parse::<IpAddr>().is_ok() {
        return Err(ToolError::Denied(
            "IP address hosts are refused; the allowlist names domains".into(),
        ));
    }
    if !allowed(&target.host, domains) {
        return Err(ToolError::Denied(format!(
            "{} is not on this server's fetch allowlist",
            target.host
        )));
    }
    Ok(target)
}

/// What [`fetch`] asks for: Markdown or HTML first (some servers answer
/// `text/markdown` themselves), then plain text, then anything.
const FETCH_ACCEPT: &str =
    "text/markdown, text/html;q=0.9, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.5";
/// Decoded HTML read per byte of the limit: Markdown keeps a fraction of a
/// page's bytes, so a page may be four times the limit before conversion.
const HTML_INPUT_FACTOR: usize = 4;

/// Fetch `url`, returning at most `max_bytes` of body text.
///
/// HTML is converted to Markdown (from at most four times `max_bytes` of
/// decoded HTML, the Markdown then capped at `max_bytes`); other text is
/// decoded from its charset; anything else is shown as UTF-8 as before.
pub async fn fetch(url: &str, domains: &[String], max_bytes: usize) -> Result<Output, ToolError> {
    let target = target(url, domains)?;
    fetch_target(&client::global()?, &target, max_bytes).await
}

async fn fetch_target(
    client: &Arc<client::Client>,
    target: &Target,
    max_bytes: usize,
) -> Result<Output, ToolError> {
    let mut response = client.get(target, FETCH_ACCEPT).await?;
    let status = response.status;
    let mut text = format!("HTTP {}\n", status.as_u16());
    for name in [http::header::CONTENT_TYPE, http::header::LOCATION] {
        if let Some(value) = response
            .headers
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
    let kind = markdown::Kind::of(
        response
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
    );
    let html = matches!(kind, markdown::Kind::Html { .. });
    let limit = if html {
        max_bytes.saturating_mul(HTML_INPUT_FACTOR)
    } else {
        max_bytes
    };
    let read = body::read(&mut response, limit).await?;
    drop(response);
    let truncated = match kind {
        markdown::Kind::Html { charset } => {
            // Milliseconds of CPU per page, off the async threads; bounded
            // by the input limit and the converter's depth limit.
            let converted =
                markdown::convert_on_thread(read.bytes, charset, url_of(target)).await?;
            text.push_str("[converted from HTML to Markdown]\n\n");
            let mut end = converted.len().min(max_bytes);
            while !converted.is_char_boundary(end) {
                end -= 1;
            }
            text.push_str(converted.get(..end).unwrap_or_default());
            if read.cut {
                let _ = write!(
                    text,
                    "\n[truncated: the page was cut at {limit} bytes of HTML before conversion]"
                );
            }
            if end < converted.len() {
                let _ = write!(
                    text,
                    "\n[truncated: the Markdown was cut at {max_bytes} bytes]"
                );
            }
            read.cut || end < converted.len()
        }
        markdown::Kind::Text { charset } => {
            text.push('\n');
            text.push_str(&markdown::decode(&read.bytes, charset.as_deref()));
            body_cut(&mut text, read.cut, max_bytes)
        }
        markdown::Kind::Other => {
            text.push('\n');
            text.push_str(&String::from_utf8_lossy(&read.bytes));
            body_cut(&mut text, read.cut, max_bytes)
        }
    };
    Ok(Output::Text { text, truncated })
}

/// Append the marker for a body cut at `max_bytes`, if it was.
fn body_cut(text: &mut String, cut: bool, max_bytes: usize) -> bool {
    if cut {
        let _ = write!(text, "\n[truncated: the body was cut at {max_bytes} bytes]");
    }
    cut
}

/// `target` as a URL, for resolving the links of its page.
fn url_of(target: &Target) -> String {
    let scheme = if target.https { "https" } else { "http" };
    format!("{scheme}://{}{}", host_header(target), target.path)
}

/// Redirects [`download`] follows.
const MAX_REDIRECTS: usize = 5;

/// A body [`download`] read whole.
#[derive(Debug)]
pub struct Download {
    /// The response's `Content-Type`, if any.
    pub content_type: Option<String>,
    pub body: Vec<u8>,
}

/// `GET` `url` from any public host, for media a request names.
///
/// A host by name or public IP literal, with the address checks of
/// [`fetch`] but no allowlist. Up to five redirects are followed, each
/// target checked anew, never from `https` to `http`. A status other than
/// success, a `Content-Type` that `accept` refuses (it gets the header's
/// value, or `None` without one; checked before any of the body is read),
/// or a body over `max_bytes`, fails. The caller's deadline bounds the whole
/// exchange.
pub async fn download(
    url: &str,
    max_bytes: usize,
    accept: impl Fn(Option<&str>) -> bool,
) -> Result<Download, ToolError> {
    download_from(&client::global()?, url, max_bytes, accept).await
}

async fn download_from(
    client: &Arc<client::Client>,
    url: &str,
    max_bytes: usize,
    accept: impl Fn(Option<&str>) -> bool,
) -> Result<Download, ToolError> {
    let mut target = parse(url)?;
    for _ in 0..=MAX_REDIRECTS {
        let mut response = client.get(&target, "*/*").await?;
        let status = response.status;
        if status.is_redirection() {
            let location = response
                .headers
                .get(http::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    ToolError::Failed(format!("HTTP {} without a Location", status.as_u16()))
                })?;
            target = redirect(&target, location)?;
            continue;
        }
        if !status.is_success() {
            return Err(ToolError::Failed(format!(
                "the server answered HTTP {}",
                status.as_u16()
            )));
        }
        let too_large = || ToolError::Invalid(format!("the body is larger than {max_bytes} bytes"));
        let headers = &response.headers;
        if headers
            .get(http::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok()?.parse::<u64>().ok())
            .is_some_and(|length| length > max_bytes as u64)
        {
            return Err(too_large());
        }
        let content_type = headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if !accept(content_type.as_deref()) {
            return Err(ToolError::Denied(format!(
                "the response's content type {} is not accepted",
                content_type.as_deref().unwrap_or("(none)")
            )));
        }
        let read = body::read(&mut response, max_bytes).await?;
        if read.cut {
            return Err(too_large());
        }
        return Ok(Download {
            content_type,
            body: read.bytes,
        });
    }
    Err(ToolError::Failed(format!(
        "more than {MAX_REDIRECTS} redirects"
    )))
}

/// The target a redirect from `base` to `location` names, refusing a
/// downgrade from `https` to `http`.
fn redirect(base: &Target, location: &str) -> Result<Target, ToolError> {
    let next = parse(&join(base, location))?;
    if base.https && !next.https {
        return Err(ToolError::Denied(
            "a redirect from https to http is never followed".into(),
        ));
    }
    Ok(next)
}

/// `location` (absolute, scheme-relative, absolute-path or relative) resolved
/// against `base`.
fn join(base: &Target, location: &str) -> String {
    let scheme = if base.https { "https" } else { "http" };
    if location.starts_with("http://") || location.starts_with("https://") {
        location.to_owned()
    } else if let Some(rest) = location.strip_prefix("//") {
        format!("{scheme}://{rest}")
    } else if location.starts_with('/') {
        format!("{scheme}://{}{location}", host_header(base))
    } else {
        let path = base.path.split(['?', '#']).next().unwrap_or_default();
        let directory = path
            .rfind('/')
            .and_then(|end| path.get(..=end))
            .unwrap_or("/");
        format!("{scheme}://{}{directory}{location}", host_header(base))
    }
}

fn host_header(target: &Target) -> String {
    let default = if target.https { 443 } else { 80 };
    let host = if target.host.contains(':') {
        format!("[{}]", target.host)
    } else {
        target.host.clone()
    };
    if target.port == default {
        host
    } else {
        format!("{host}:{}", target.port)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn redirect_locations_resolve_against_the_request() {
        let base = super::parse("https://cdn.example.com:8443/media/a/b.png?x=1").expect("url");
        for (location, expected) in [
            ("http://other.example/x.png", "http://other.example/x.png"),
            ("//img.example/y.png", "https://img.example/y.png"),
            ("/z.png", "https://cdn.example.com:8443/z.png"),
            (
                "c.png?v=2",
                "https://cdn.example.com:8443/media/a/c.png?v=2",
            ),
        ] {
            assert_eq!(super::join(&base, location), expected);
        }
        let v6 = super::parse("http://[2606:4700::1]/p/q").expect("url");
        assert_eq!(v6.host, "2606:4700::1");
        assert_eq!(super::join(&v6, "r"), "http://[2606:4700::1]/p/r");
    }

    #[test]
    fn redirects_never_downgrade_https_to_http() {
        let secure = super::parse("https://cdn.example.com/a.png").expect("url");
        assert!(matches!(
            super::redirect(&secure, "http://cdn.example.com/a.png"),
            Err(super::ToolError::Denied(_))
        ));
        assert!(
            super::redirect(&secure, "/b.png")
                .expect("same scheme")
                .https
        );
        let plain = super::parse("http://cdn.example.com/a.png").expect("url");
        assert!(
            super::redirect(&plain, "https://cdn.example.com/a.png")
                .expect("upgrade")
                .https
        );
    }
}
