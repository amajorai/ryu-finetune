//! Remote GPU node base-URL validation and the pinned, SSRF-safe client.

use super::*;

pub(super) fn disallowed_remote_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, ..] = ip.octets();
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_broadcast()
                || a == 0
                || a == 10
                || (a == 100 && (64..=127).contains(&b))
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 168)
                || (a == 192 && b == 0)
                || (a == 198 && (18..=19).contains(&b))
                || a >= 224
        }
        IpAddr::V6(ip) => {
            if let Some(ipv4) = ip.to_ipv4() {
                return disallowed_remote_ip(IpAddr::V4(ipv4));
            }
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

pub(super) fn remote_origin(url: &reqwest::Url) -> String {
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host
    };
    let mut origin = format!("{}://{host}", url.scheme());
    if let Some(port) = url.port() {
        origin.push(':');
        origin.push_str(&port.to_string());
    }
    origin
}

pub(super) fn normalize_remote_base_url(raw: &str) -> Result<String, String> {
    let candidate = raw.trim().trim_end_matches('/');
    if candidate.is_empty() || candidate.chars().count() > MAX_REMOTE_URL_CHARS {
        return Err(format!(
            "remote.url must be a non-empty URL of at most {MAX_REMOTE_URL_CHARS} characters"
        ));
    }
    let url = reqwest::Url::parse(candidate)
        .map_err(|_| "remote.url must be a valid absolute URL".to_owned())?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("remote.url must not contain userinfo".to_owned());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("remote.url must not contain a query or fragment".to_owned());
    }
    if url.path() != "" && url.path() != "/" {
        return Err("remote.url must be a bare origin without a path".to_owned());
    }
    let host = url
        .host_str()
        .map(str::to_ascii_lowercase)
        .filter(|host| !host.is_empty())
        .ok_or_else(|| "remote.url must contain a host".to_owned())?;
    let ip_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(&host);
    let loopback_dev = cfg!(debug_assertions)
        && url.scheme() == "http"
        && ip_host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if url.scheme() != "https" && !loopback_dev {
        return Err("remote.url must use https".to_owned());
    }
    if host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host == "metadata.google.internal"
        || host == "metadata"
    {
        return Err("remote.url must not target a local or metadata host".to_owned());
    }
    if let Ok(ip) = ip_host.parse::<IpAddr>() {
        if disallowed_remote_ip(ip) && !loopback_dev {
            return Err(
                "remote.url must not target a private, loopback, or link-local address".to_owned(),
            );
        }
    }
    Ok(remote_origin(&url))
}

pub(super) fn validate_remote_base_url(raw: &str) -> Result<String, String> {
    let origin = normalize_remote_base_url(raw)?;
    if let Ok(configured) = std::env::var(REMOTE_ALLOWLIST_ENV) {
        let configured = configured.trim();
        if !configured.is_empty()
            && !configured
                .split(',')
                .filter_map(|entry| normalize_remote_base_url(entry).ok())
                .any(|allowed| allowed == origin)
        {
            return Err(format!(
                "remote.url is not in the configured {REMOTE_ALLOWLIST_ENV} allowlist"
            ));
        }
    }
    Ok(origin)
}

/// Resolve a remote hostname once, reject private answers, and pin the safe
/// answer into a redirect-disabled client. This closes the DNS-rebinding gap
/// left by validating only the URL text: every proxy request uses the same
/// public address that passed the check, and a redirect cannot move the bearer
/// to another host.
pub(super) async fn remote_client_for_base(
    ctx: &FinetuneCtx,
    base: &str,
) -> Result<reqwest::Client, String> {
    let url = reqwest::Url::parse(base)
        .map_err(|_| "remote.url must be a valid absolute URL".to_owned())?;
    let host = url
        .host_str()
        .ok_or_else(|| "remote.url must contain a host".to_owned())?;
    let ip_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if ip_host.parse::<IpAddr>().is_ok() {
        return Ok(ctx.remote_client.clone());
    }

    let port = url.port_or_known_default().unwrap_or(443);
    let mut addresses = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        tokio::net::lookup_host((host, port)),
    )
    .await
    .map_err(|_| "remote hostname resolution timed out".to_owned())?
    .map_err(|_| "remote hostname could not be resolved".to_owned())?;
    let address: SocketAddr = addresses
        .find(|address| !disallowed_remote_ip(address.ip()))
        .ok_or_else(|| "remote hostname resolved only to private or local addresses".to_owned())?;

    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        .resolve(host, address)
        .build()
        .map_err(|_| "safe remote HTTP client could not be created".to_owned())
}
