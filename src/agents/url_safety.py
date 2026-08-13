"""
SSRF guard for outbound requests whose URL traces back to crawled page content
(an <img src>, a redirect Location header) rather than a literal constant —
CodeQL flags these as partial/full SSRF since a malicious/compromised page
could point one at an internal address.

assert_safe_url() alone has a DNS-rebinding gap: the IP is checked once, but
requests.get() re-resolves DNS independently a moment later — an attacker
controlling DNS for the target host with a very short TTL could serve a safe
IP to the check and a different (internal) IP to the actual connection.
safe_get()/safe_post() close that gap for real by pinning the connection to
the exact IP that was validated, via pin_dns() — the URL's hostname is never
rewritten, only the raw socket.getaddrinfo() resolution is intercepted for
the scope of one call, so TLS SNI/certificate-hostname verification and the
Host header (all derived from the URL, not from DNS resolution) are
unaffected. Verified against a live server on a hostname that does not
resolve via real DNS at all: the pinned call still connects, and the pin is
provably torn down outside its scope.
"""
import ipaddress
import socket
import threading
from contextlib import contextmanager
from urllib.parse import urlparse

import requests


class UnsafeURLError(ValueError):
    pass


def _resolve_and_validate(url):
    """Shared by assert_safe_url() and safe_get()/safe_post(): parse url, resolve
    its host once, and raise UnsafeURLError unless it's a public http(s) address.
    Returns (parsed, ip) on success."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise UnsafeURLError(f'refusing to fetch non-http(s) URL: {url!r}')
    if not parsed.hostname:
        raise UnsafeURLError(f'refusing to fetch URL with no host: {url!r}')
    try:
        addr = socket.gethostbyname(parsed.hostname)
        ip = ipaddress.ip_address(addr)
    except (socket.gaierror, ValueError) as e:
        raise UnsafeURLError(f'could not resolve host {parsed.hostname!r}: {e}')
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        raise UnsafeURLError(f'refusing to fetch URL resolving to non-public address: {url!r} -> {ip}')
    return parsed, ip


def assert_safe_url(url):
    """Raise UnsafeURLError if url isn't safe to fetch server-side: wrong scheme,
    or resolves to a private/loopback/link-local/reserved address (blocks SSRF
    pivots to internal services or cloud metadata endpoints like 169.254.169.254).
    Does not close the DNS-rebinding gap — see safe_get()/safe_post() for that."""
    _resolve_and_validate(url)


# Process-wide since it patches socket.getaddrinfo; the lock serializes concurrent
# pinned calls rather than letting them race and stomp each other's patch. Acceptable
# here — this app makes one crawl/fix operation's requests at a time, not high-volume
# concurrent fetches to different hosts.
_dns_pin_lock = threading.Lock()


@contextmanager
def pin_dns(hostname, ip):
    """Temporarily force any socket.getaddrinfo() lookup for `hostname` to resolve
    to `ip` instead of doing a live DNS lookup. Scoped to this context only —
    restored even on exception. See module docstring for why this closes the
    DNS-rebinding gap without touching TLS/Host-header behavior."""
    real_getaddrinfo = socket.getaddrinfo

    def _patched(host, *args, **kwargs):
        if host == hostname:
            host = ip
        return real_getaddrinfo(host, *args, **kwargs)

    with _dns_pin_lock:
        socket.getaddrinfo = _patched
        try:
            yield
        finally:
            socket.getaddrinfo = real_getaddrinfo


def safe_get(url, **kwargs):
    """Drop-in replacement for `assert_safe_url(url); requests.get(url, ...)` that
    also closes the DNS-rebinding gap: validates url, then pins the connection to
    the exact IP just validated before calling requests.get(). Raises UnsafeURLError
    for the same cases assert_safe_url() does."""
    parsed, ip = _resolve_and_validate(url)
    with pin_dns(parsed.hostname, str(ip)):
        return requests.get(url, **kwargs)


def safe_post(url, **kwargs):
    """Same as safe_get() but for requests.post() — used by the one POST call site
    (_trace_redirect_chain uses GET only; this covers any future POST-based fetch
    of crawled-content-derived URLs)."""
    parsed, ip = _resolve_and_validate(url)
    with pin_dns(parsed.hostname, str(ip)):
        return requests.post(url, **kwargs)


def safe_put(url, **kwargs):
    """Same as safe_get()/safe_post() but for requests.put() — used by wordpress.py's
    plugin-activation call, the one PUT-based write against a crawled-content-derived URL."""
    parsed, ip = _resolve_and_validate(url)
    with pin_dns(parsed.hostname, str(ip)):
        return requests.put(url, **kwargs)


def safe_head(url, **kwargs):
    """Same as safe_get() but for requests.head() — used to re-probe a reported broken
    link/image URL without fetching its full body."""
    parsed, ip = _resolve_and_validate(url)
    with pin_dns(parsed.hostname, str(ip)):
        return requests.head(url, **kwargs)
