//! Protection from web pages: what a local browser may send this server.
//!
//! Authentication is off by default and the server binds `127.0.0.1`, so
//! without this any web page a local browser opens could use it, in two
//! ways (the attacks Ollama fixed in CVE-2024-28224 and llama.cpp limits
//! with `--cors-origins`):
//!
//! * **DNS rebinding.** A page on `evil.example` re-points its name at
//!   `127.0.0.1` (or the machine's LAN address) and reads this server as its
//!   own origin. The browser still sends `Host: evil.example`, so a request is
//!   refused (403) unless its `Host` (or HTTP/2 and HTTP/3 `:authority`) is
//!   `localhost`, a `*.localhost` name, a `*.internal` name (Docker's
//!   `host.docker.internal`; the TLD is reserved for private use, so public
//!   DNS cannot rebind it), a `*.local` name (resolved by multicast DNS on the
//!   local network, never public DNS), an IP address (rebinding needs a name),
//!   or a name given with `--allow-host`. A request without one (HTTP/1.0) is
//!   not from a browser and passes. While bound to a loopback address this
//!   always applies; bound to another address it applies while
//!   authentication is off, since a page that rebinds a name has no key.
//! * **Cross-site requests.** A page can send requests it cannot read the
//!   answer to (a `text/plain` POST needs no CORS preflight), which is
//!   enough to start generations or delete stored objects. A request whose
//!   `Origin` is another site, or that carries no `Origin` but
//!   `Sec-Fetch-Site: cross-site`, is refused (403) while authentication is
//!   off. Allowed without configuration: pages on loopback hosts (`http(s)`
//!   on `localhost`, `*.localhost`, `127.0.0.0/8`, `[::1]`, any port, as
//!   Ollama's default origins), and origins that are not `http(s)` at all
//!   (`tauri://`, `vscode-webview://`, `chrome-extension://`, ...), which only
//!   local applications and extensions have. `null` (sandboxed frames,
//!   `file:` pages) is a web origin anyone can produce and is refused.
//!   `--allow-origin ORIGIN` (repeatable; `*` for every origin, `null` for
//!   `file:` pages) allows more. Once authentication is on, an authorized
//!   request passes whatever its origin, since a page holding a key was
//!   given it.
//!
//! CLIs and SDKs send no `Origin` or `Sec-Fetch-*` headers and address the
//! server as `localhost` or an IP, so neither check ever affects them.
//!
//! An admitted request with an `Origin` gets CORS headers reflecting it, and
//! a CORS preflight (`OPTIONS` with `Access-Control-Request-Method`) from an
//! admitted origin is answered `204` here, so browser applications can read
//! the answers (authentication on: every origin, as requests still need a
//! key).

use std::net::IpAddr;

use axum::body::Body;
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::Response;

/// Seconds a browser may cache a preflight (Chromium caps it at two hours).
const PREFLIGHT_MAX_AGE: &str = "7200";

/// Which browser requests this server admits.
#[derive(Debug)]
pub(super) struct Browsers {
    /// Whether the server is bound to a loopback address, where `Host` is
    /// checked.
    loopback: bool,
    /// `--allow-origin` values, exactly as given.
    allowed: Vec<String>,
    /// `--allow-host` names, normalized as [`host_name`] does.
    hosts: Vec<String>,
}

/// A request refused for its `Host` (DNS rebinding protection).
#[derive(Debug)]
pub(super) struct HostRefusal {
    pub(super) message: String,
    /// Whether authentication being on admits the request anyway: bound to
    /// a non-loopback address, where a page that rebinds a name holds no key.
    pub(super) unless_authenticated: bool,
}

/// A browser request refused while authentication is off.
#[derive(Debug)]
pub(super) struct Foreign {
    /// The request's `Origin`, reflected when authentication admits it
    /// anyway.
    pub(super) origin: Option<HeaderValue>,
    pub(super) message: String,
}

impl Browsers {
    pub(super) const fn new(host: IpAddr, allowed: Vec<String>, hosts: Vec<String>) -> Self {
        Self {
            loopback: host.is_loopback(),
            allowed,
            hosts,
        }
    }

    /// Why the request's host is refused (DNS rebinding), if it is.
    pub(super) fn host_refusal(&self, parts: &Parts) -> Option<HostRefusal> {
        let authority = match parts.uri.authority() {
            Some(authority) => authority.as_str(),
            None => parts
                .headers
                .get(header::HOST)?
                .to_str()
                .unwrap_or("\u{fffd}"),
        };
        let admitted = local_host(authority)
            || host_name(authority).is_some_and(|name| self.hosts.contains(&name));
        if admitted {
            return None;
        }
        Some(if self.loopback {
            HostRefusal {
                message: format!(
                    "host {authority:?} is not a loopback name or address; this server is bound \
                     to loopback and answers only requests addressed to localhost or an IP \
                     address, or to a name given with --allow-host (protection from DNS \
                     rebinding)"
                ),
                unless_authenticated: false,
            }
        } else {
            HostRefusal {
                message: format!(
                    "host {authority:?} is not an IP address or a local name; while \
                     authentication is off this server answers only requests addressed to an IP \
                     address, localhost or a *.local or *.internal name, or to a name given with \
                     --allow-host {} (protection from DNS rebinding)",
                    host_name(authority).unwrap_or_default()
                ),
                unless_authenticated: true,
            }
        })
    }

    /// The request's `Origin` to reflect when it is admitted (`None` without
    /// one), or why it is refused while authentication is off.
    pub(super) fn origin(&self, headers: &HeaderMap) -> Result<Option<HeaderValue>, Foreign> {
        let any = self.allowed.iter().any(|allowed| allowed == "*");
        let Some(origin) = headers.get(header::ORIGIN) else {
            let cross_site = headers
                .get("sec-fetch-site")
                .is_some_and(|site| site.as_bytes().eq_ignore_ascii_case(b"cross-site"));
            return if cross_site && !any {
                Err(Foreign {
                    origin: None,
                    message: "cross-site browser requests are refused while authentication is \
                              off (Sec-Fetch-Site: cross-site); see --allow-origin"
                        .into(),
                })
            } else {
                Ok(None)
            };
        };
        let text = origin.to_str().unwrap_or_default();
        if any || local_origin(text) || self.allowed.iter().any(|allowed| allowed == text) {
            return Ok(Some(origin.clone()));
        }
        Err(Foreign {
            origin: Some(origin.clone()),
            message: format!(
                "browser requests from origin {text:?} are refused while authentication is off; \
                 allow it with --allow-origin {text}, or turn authentication on"
            ),
        })
    }
}

/// Whether `authority` (`host[:port]`) names this machine in a way public
/// DNS cannot rebind.
fn local_host(authority: &str) -> bool {
    if let Some(rest) = authority.strip_prefix('[') {
        // `[v6]:port`: an IP address.
        return rest
            .split_once(']')
            .is_some_and(|(v6, _)| v6.parse::<IpAddr>().is_ok());
    }
    host_name(authority).is_some_and(|host| {
        host.parse::<IpAddr>().is_ok()
            || host == "localhost"
            || host.ends_with(".localhost")
            || host.ends_with(".internal")
            || host
                .strip_suffix(".local")
                .is_some_and(|name| !name.is_empty())
    })
}

/// The host name in `authority` (`host[:port]`), lowercase and without a
/// trailing dot; `None` for a bracketed IPv6 address.
pub(super) fn host_name(authority: &str) -> Option<String> {
    if authority.starts_with('[') {
        return None;
    }
    let host = authority
        .rsplit_once(':')
        .map_or(authority, |(host, _port)| host);
    Some(host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase())
}

/// Whether a browser `Origin` is admitted without configuration: an
/// `http(s)` page on a loopback host, or a non-web scheme.
fn local_origin(origin: &str) -> bool {
    let Some((scheme, rest)) = origin.split_once("://") else {
        // `null`, or not an origin at all.
        return false;
    };
    if !matches!(scheme.to_ascii_lowercase().as_str(), "http" | "https") {
        return true;
    }
    let host = match rest.strip_prefix('[') {
        Some(rest) => {
            return rest
                .split_once(']')
                .and_then(|(v6, _)| v6.parse::<IpAddr>().ok())
                .is_some_and(|address| address.is_loopback());
        }
        None => rest.rsplit_once(':').map_or(rest, |(host, _port)| host),
    };
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host.ends_with(".localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

/// Whether `parts` is a CORS preflight.
pub(super) fn is_preflight(parts: &Parts) -> bool {
    parts.method == Method::OPTIONS
        && parts.headers.contains_key(header::ORIGIN)
        && parts
            .headers
            .contains_key(header::ACCESS_CONTROL_REQUEST_METHOD)
}

/// The `204` answering an admitted preflight from `origin` (which a
/// preflight always has): every method and header it asks for (the route
/// itself decides what they mean).
pub(super) fn preflight(origin: Option<HeaderValue>, request: &HeaderMap) -> Response {
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::NO_CONTENT;
    let headers = response.headers_mut();
    for (name, asked) in [
        (
            header::ACCESS_CONTROL_ALLOW_METHODS,
            header::ACCESS_CONTROL_REQUEST_METHOD,
        ),
        (
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            header::ACCESS_CONTROL_REQUEST_HEADERS,
        ),
    ] {
        if let Some(value) = request.get(asked) {
            headers.insert(name, value.clone());
        }
    }
    headers.insert(
        header::ACCESS_CONTROL_MAX_AGE,
        HeaderValue::from_static(PREFLIGHT_MAX_AGE),
    );
    headers.append(
        header::VARY,
        HeaderValue::from_static("Access-Control-Request-Method, Access-Control-Request-Headers"),
    );
    if let Some(origin) = origin {
        allow(&mut response, origin);
    }
    response
}

/// Let `origin` read `response`.
pub(super) fn allow(response: &mut Response, origin: HeaderValue) {
    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    headers.insert(
        header::ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static("*"),
    );
    headers.append(header::VARY, HeaderValue::from_static("Origin"));
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use axum::http::Request;

    fn parts(uri: &str, headers: &[(&str, &str)]) -> Parts {
        let mut request = Request::builder().uri(uri);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        request.body(()).expect("request").into_parts().0
    }

    fn loopback(allowed: &[&str]) -> Browsers {
        Browsers::new(
            "127.0.0.1".parse().expect("ip"),
            allowed.iter().map(|&origin| origin.to_owned()).collect(),
            Vec::new(),
        )
    }

    #[test]
    fn loopback_binding_admits_only_local_hosts() {
        let browsers = loopback(&[]);
        for host in [
            "127.0.0.1:8080",
            "localhost:8080",
            "LOCALHOST",
            "localhost.:8080",
            "bonsai.localhost:8080",
            "host.docker.internal:8080",
            "studio.local:8080",
            "[::1]:8080",
            "0.0.0.0:8080",
        ] {
            let request = parts("/v1/models", &[("host", host)]);
            assert!(browsers.host_refusal(&request).is_none(), "{host}");
        }
        for host in [
            "evil.example:8080",
            "evil.example",
            "localhost.evil.example",
            "127.0.0.1.nip.io:8080",
            "[::1",
        ] {
            let request = parts("/v1/models", &[("host", host)]);
            let refusal = browsers.host_refusal(&request).expect(host);
            assert!(refusal.message.contains("DNS rebinding"), "{refusal:?}");
            assert!(!refusal.unless_authenticated, "always refused on loopback");
        }
        // HTTP/2 and HTTP/3 carry the host in the URI (`:authority`).
        let request = parts("http://evil.example:8080/v1/models", &[]);
        assert!(browsers.host_refusal(&request).is_some());
        // HTTP/1.0 without a Host is no browser.
        assert!(browsers.host_refusal(&parts("/v1/models", &[])).is_none());
    }

    #[test]
    fn a_non_loopback_binding_refuses_rebound_names_while_authentication_is_off() {
        let open = Browsers::new(
            "0.0.0.0".parse().expect("ip"),
            Vec::new(),
            vec!["ai.example.com".to_owned()],
        );
        // After rebinding, the page's own name arrives: refused, unless
        // authentication is on (the page has no key).
        let request = parts("/v1/files/file-1/content", &[("host", "evil.example:8080")]);
        let refusal = open.host_refusal(&request).expect("rebound name refused");
        assert!(refusal.unless_authenticated);
        assert!(
            refusal.message.contains("--allow-host evil.example"),
            "{refusal:?}"
        );
        // LAN clients by address or local name, and allowed names, pass.
        for host in [
            "192.168.1.20:8080",
            "[fe80::1]:8080",
            "localhost:8080",
            "studio.local:8080",
            "host.docker.internal",
            "AI.example.com.:8080",
        ] {
            let request = parts("/v1/models", &[("host", host)]);
            assert!(open.host_refusal(&request).is_none(), "{host}");
        }
        // The allowance is exact: a subdomain is another name.
        let request = parts("/v1/models", &[("host", "x.ai.example.com")]);
        assert!(open.host_refusal(&request).is_some());
    }

    #[test]
    fn cross_site_origins_are_refused_unless_allowed() {
        let browsers = loopback(&["https://chat.example.com"]);
        let origin = |value: &str| browsers.origin(&parts("/", &[("origin", value)]).headers);
        for local in [
            "http://localhost:3000",
            "https://localhost",
            "http://app.localhost:5173",
            "http://127.0.0.1:8080",
            "http://127.1.2.3",
            "http://[::1]:3000",
            "tauri://localhost",
            "vscode-webview://1a2b3c",
            "chrome-extension://abcdefghijklmnop",
            "https://chat.example.com",
        ] {
            let reflected = origin(local).expect(local).expect(local);
            assert_eq!(reflected, local);
        }
        for foreign in [
            "https://evil.example",
            "http://localhost.evil.example",
            "http://192.168.1.5:3000",
            "https://chat.example.com:8443",
            "null",
        ] {
            let refused = origin(foreign).expect_err(foreign);
            assert_eq!(refused.origin.expect(foreign), foreign);
        }
        // No Origin: Sec-Fetch-Site decides, and CLIs send neither.
        let site = |value: &str| browsers.origin(&parts("/", &[("sec-fetch-site", value)]).headers);
        assert!(site("cross-site").is_err());
        for admitted in ["same-origin", "same-site", "none"] {
            assert_eq!(site(admitted).expect(admitted), None);
        }
        assert_eq!(browsers.origin(&HeaderMap::new()).expect("cli"), None);
    }

    #[test]
    fn a_wildcard_or_null_entry_allows_those_origins() {
        let any = loopback(&["*"]);
        let headers = parts("/", &[("origin", "https://evil.example")]).headers;
        assert!(any.origin(&headers).is_ok());
        let headers = parts("/", &[("sec-fetch-site", "cross-site")]).headers;
        assert!(any.origin(&headers).is_ok());
        let null = loopback(&["null"]);
        let headers = parts("/", &[("origin", "null")]).headers;
        assert!(null.origin(&headers).is_ok());
    }

    #[test]
    fn preflights_reflect_what_they_ask_for() {
        let request = parts(
            "/v1/chat/completions",
            &[
                ("origin", "http://localhost:3000"),
                ("access-control-request-method", "POST"),
                (
                    "access-control-request-headers",
                    "authorization, content-type",
                ),
            ],
        );
        let mut request = request;
        request.method = Method::OPTIONS;
        assert!(is_preflight(&request));
        let response = preflight(
            Some(HeaderValue::from_static("http://localhost:3000")),
            &request.headers,
        );
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        let headers = response.headers();
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "http://localhost:3000"
        );
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_METHODS], "POST");
        assert_eq!(
            headers[header::ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization, content-type"
        );
        // A plain OPTIONS is no preflight.
        let mut plain = parts("/v1/models", &[("origin", "http://localhost:3000")]);
        plain.method = Method::OPTIONS;
        assert!(!is_preflight(&plain));
    }
}
