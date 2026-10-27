//! Which origins speak HTTP/3, learnt from `Alt-Svc` response headers and
//! DNS HTTPS records, and which have recently failed over QUIC.
//!
//! Only an alternative on the origin's own host is kept (a different port
//! is fine): the QUIC connection then goes to the addresses already resolved
//! and checked for that host, with the same name verified in the
//! certificate, so an alternative can never steer a fetch to another host or
//! an unchecked address. An alternative naming another host is ignored.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::client::Key;

/// `Alt-Svc` freshness when the header gives no `ma` (RFC 7838).
const DEFAULT_MAX_AGE: Duration = Duration::from_hours(24);
/// Longest freshness kept, whatever the header or record says.
const MAX_AGE_CAP: Duration = Duration::from_hours(7 * 24);
/// How long an origin is kept off QUIC after its first failure; each
/// further failure doubles it, up to [`BROKEN_CAP`] (as browsers do).
const BROKEN_BASE: Duration = Duration::from_mins(5);
const BROKEN_CAP: Duration = Duration::from_hours(48);
/// How long a DNS HTTPS lookup that found nothing is not repeated.
const HTTPS_RR_NEGATIVE: Duration = Duration::from_mins(5);
/// Origins remembered in each table.
const MAX_ENTRIES: usize = 256;

/// What one `Alt-Svc` header value says about HTTP/3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advert {
    /// `clear`: forget the origin's alternatives.
    Clear,
    /// The origin's host serves HTTP/3 on `port` for `max_age`.
    H3 { port: u16, max_age: Duration },
}

/// Parse an `Alt-Svc` value for an origin on `host`: its first `h3`
/// alternative on the same host (any other host is ignored), or `clear`.
#[must_use]
pub fn parse(value: &str, host: &str) -> Option<Advert> {
    if value.trim().eq_ignore_ascii_case("clear") {
        return Some(Advert::Clear);
    }
    split_outside_quotes(value, ',').find_map(|alternative| {
        let mut parts = split_outside_quotes(alternative, ';');
        let (protocol, authority) = parts.next()?.split_once('=')?;
        if protocol.trim() != "h3" {
            return None;
        }
        let authority = authority.trim().strip_prefix('"')?.strip_suffix('"')?;
        let (alt_host, port) = authority.rsplit_once(':')?;
        let alt_host = alt_host.trim_start_matches('[').trim_end_matches(']');
        if !(alt_host.is_empty() || alt_host.trim_end_matches('.').eq_ignore_ascii_case(host)) {
            return None;
        }
        let port = port.parse::<u16>().ok().filter(|port| *port != 0)?;
        let max_age = parts
            .filter_map(|parameter| parameter.split_once('='))
            .find(|(name, _)| name.trim() == "ma")
            .map_or(Some(DEFAULT_MAX_AGE), |(_, seconds)| {
                seconds.trim().parse::<u64>().ok().map(Duration::from_secs)
            })?;
        Some(Advert::H3 {
            port,
            max_age: max_age.min(MAX_AGE_CAP),
        })
    })
}

/// `value` split at `separator`s that are not inside a quoted string.
fn split_outside_quotes(value: &str, separator: char) -> impl Iterator<Item = &str> {
    let mut quoted = false;
    value
        .split(move |c: char| {
            if c == '"' {
                quoted = !quoted;
            }
            c == separator && !quoted
        })
        .filter(|part| !part.trim().is_empty())
}

#[derive(Debug, Default)]
struct Tables {
    /// HTTP/3 port and expiry per origin.
    alternatives: HashMap<Key, (u16, Instant)>,
    /// Until when, and after how many failures, an origin stays off QUIC.
    broken: HashMap<Key, (Instant, u32)>,
    /// Until when an origin's DNS HTTPS record needs no new lookup.
    looked_up: HashMap<Key, Instant>,
}

/// The shared cache of HTTP/3 alternatives.
#[derive(Debug, Default)]
pub struct Cache {
    tables: Mutex<Tables>,
}

impl Cache {
    fn tables(&self) -> std::sync::MutexGuard<'_, Tables> {
        self.tables.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The port to try HTTP/3 on for `key`, when it is known to serve it
    /// and has not failed recently.
    pub fn h3_port(&self, key: &Key) -> Option<u16> {
        let now = Instant::now();
        let tables = self.tables();
        if tables
            .broken
            .get(key)
            .is_some_and(|(until, _)| *until > now)
        {
            return None;
        }
        tables
            .alternatives
            .get(key)
            .filter(|(_, expires)| *expires > now)
            .map(|(port, _)| *port)
    }

    /// Record the `Alt-Svc` header values of a response from `key`.
    pub fn learn<'a>(&self, key: &Key, values: impl Iterator<Item = &'a str>) {
        for value in values {
            match parse(value, &key.host) {
                Some(Advert::Clear) => {
                    self.tables().alternatives.remove(key);
                    return;
                }
                Some(Advert::H3 { port, max_age }) => {
                    self.advertise(key, port, max_age);
                    return;
                }
                None => {}
            }
        }
    }

    /// Record that `key`'s host serves HTTP/3 on `port` for `max_age`. As
    /// in browsers, an origin on an unprivileged port cannot point at a
    /// privileged one (below 1024).
    pub fn advertise(&self, key: &Key, port: u16, max_age: Duration) {
        if port == 0 || max_age.is_zero() || (port < 1024 && key.port >= 1024) {
            return;
        }
        let now = Instant::now();
        let mut tables = self.tables();
        bound(&mut tables.alternatives, now, |(_, expires)| *expires);
        tables
            .alternatives
            .insert(key.clone(), (port, now + max_age.min(MAX_AGE_CAP)));
    }

    /// Keep `key` off QUIC for a while after a failed QUIC attempt.
    pub fn mark_broken(&self, key: &Key) {
        let now = Instant::now();
        let mut tables = self.tables();
        let failures = tables.broken.get(key).map_or(0, |(_, failures)| *failures) + 1;
        let backoff = BROKEN_BASE
            .saturating_mul(1 << (failures - 1).min(16))
            .min(BROKEN_CAP);
        bound(&mut tables.broken, now, |(until, _)| *until);
        tables.broken.insert(key.clone(), (now + backoff, failures));
    }

    /// Forget `key`'s QUIC failures after a QUIC connection succeeded.
    pub fn confirm(&self, key: &Key) {
        self.tables().broken.remove(key);
    }

    /// Whether `key`'s DNS HTTPS record should be looked up now; marks it
    /// looked up for a while, so concurrent and later fetches do not repeat
    /// the query. [`Cache::looked_up`] then records how long the answer
    /// lasts.
    pub fn https_rr_due(&self, key: &Key) -> bool {
        let now = Instant::now();
        let mut tables = self.tables();
        if tables.looked_up.get(key).is_some_and(|until| *until > now) {
            return false;
        }
        bound(&mut tables.looked_up, now, |until| *until);
        tables
            .looked_up
            .insert(key.clone(), now + HTTPS_RR_NEGATIVE);
        true
    }

    /// Record a DNS HTTPS answer for `key`: HTTP/3 on `port` for `ttl`, or
    /// nothing.
    pub fn looked_up(&self, key: &Key, h3: Option<(u16, Duration)>) {
        if let Some((port, ttl)) = h3 {
            self.advertise(key, port, ttl);
            let until = Instant::now() + ttl.clamp(Duration::from_secs(1), MAX_AGE_CAP);
            self.tables().looked_up.insert(key.clone(), until);
        }
    }
}

/// Make room for one more entry: drop expired ones, then, if still full, an
/// arbitrary one.
fn bound<V>(table: &mut HashMap<Key, V>, now: Instant, expiry: impl Fn(&V) -> Instant) {
    if table.len() < MAX_ENTRIES {
        return;
    }
    table.retain(|_, value| expiry(value) > now);
    if table.len() >= MAX_ENTRIES
        && let Some(key) = table.keys().next().cloned()
    {
        table.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Advert, Cache, parse};
    use crate::tools::fetch::client::Key;

    #[test]
    fn alt_svc_keeps_only_same_host_h3() {
        let day = Duration::from_hours(24);
        for (value, expected) in [
            (r#"h3=":443"; ma=86400"#, Some((443, day))),
            (r#"h3=":8443""#, Some((8443, day))),
            (
                r#"h3-29=":443"; ma=60, h3=":443"; ma=3600"#,
                Some((443, Duration::from_secs(3600))),
            ),
            (
                r#"h3="fetch.test:4433"; ma=10"#,
                Some((4433, Duration::from_secs(10))),
            ),
            (r#"h3="FETCH.TEST.:4433""#, Some((4433, day))),
            // Another host, a zero or missing port, or not h3: ignored.
            (r#"h3="evil.example:443""#, None),
            (r#"h3="127.0.0.1:443""#, None),
            (r#"h3=":0""#, None),
            (r#"h3=":""#, None),
            (r#"h2=":443""#, None),
            (r#"h3=":443"; ma=soon"#, None),
            ("", None),
        ] {
            let parsed = parse(value, "fetch.test").map(|advert| match advert {
                Advert::H3 { port, max_age } => (port, max_age),
                Advert::Clear => (0, Duration::ZERO),
            });
            assert_eq!(parsed, expected, "{value}");
        }
        // A quoted comma does not split alternatives; the first usable one
        // wins; a cap applies to very long lifetimes.
        assert_eq!(
            parse(
                r#"h3="evil,example:1", h3=":443"; ma=99999999"#,
                "fetch.test"
            ),
            Some(Advert::H3 {
                port: 443,
                max_age: Duration::from_hours(7 * 24)
            })
        );
        assert_eq!(parse(" clear ", "fetch.test"), Some(Advert::Clear));
    }

    #[test]
    fn broken_origins_stay_off_quic_and_clear_resets() {
        let cache = Cache::default();
        let key = Key::new(true, "fetch.test", 443);
        assert_eq!(cache.h3_port(&key), None);
        cache.learn(&key, [r#"h3="other.test:1""#, r#"h3=":4433""#].into_iter());
        assert_eq!(cache.h3_port(&key), Some(4433));
        cache.mark_broken(&key);
        assert_eq!(cache.h3_port(&key), None);
        cache.confirm(&key);
        assert_eq!(cache.h3_port(&key), Some(4433));
        cache.learn(&key, std::iter::once("clear"));
        assert_eq!(cache.h3_port(&key), None);
        assert!(cache.https_rr_due(&key));
        assert!(!cache.https_rr_due(&key));
        cache.looked_up(&key, Some((443, Duration::from_secs(300))));
        assert_eq!(cache.h3_port(&key), Some(443));
        // An origin on a high port cannot point at a privileged one.
        let high = Key::new(true, "fetch.test", 8443);
        cache.learn(&high, std::iter::once(r#"h3=":25""#));
        assert_eq!(cache.h3_port(&high), None);
        cache.learn(&high, std::iter::once(r#"h3=":8444""#));
        assert_eq!(cache.h3_port(&high), Some(8444));
    }
}
