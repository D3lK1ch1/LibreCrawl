import os
import re
import requests
from urllib.parse import urlparse, parse_qs, urlunparse

from src.agents.url_safety import safe_get, safe_post, safe_put


# Known REST namespace -> WordPress.org plugin slug, for translating a site's
# public /wp-json/ namespace list into an installable plugin list.
NAMESPACE_PLUGIN_MAP = {
    'rankmath/v1': 'seo-by-rank-math',
    'yoast/v1': 'wordpress-seo',
    'wpseo/v1': 'wordpress-seo',
    'redirection/v1': 'redirection',
    'wc/v3': 'woocommerce',
    'contact-form-7/v1': 'contact-form-7',
    'wp-rocket/v1': 'wp-rocket',
}


def _base_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def get_site_config(url):
    """Return Agent 3 WordPress access config for a crawled URL.

    WP_PUBLIC_URL is the URL seen by LibreCrawl/Azure tickets. WP_URL is the
    WordPress URL Agent 3 should use for REST. This lets tickets use one
    public/staging URL while Agent 3 targets a different internal service URL if
    needed.
    """
    crawled_base = _base_url(url).rstrip('/')
    public_base = os.getenv('WP_PUBLIC_URL', '').rstrip('/')
    wp_base = os.getenv('WP_URL', crawled_base).rstrip('/')

    if public_base and crawled_base == public_base:
        target_base = wp_base
    else:
        target_base = crawled_base

    return {
        'crawled_base': crawled_base,
        'public_base': public_base,
        'target_base': target_base,
    }


def map_to_target_url(url):
    config = get_site_config(url)
    parsed = urlparse(url)
    target = urlparse(config['target_base'])
    return urlunparse((target.scheme, target.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _wp_auth():
    username = os.getenv('WP_USERNAME')
    app_password = os.getenv('WP_APP_PASSWORD')
    if username and app_password:
        return (username, app_password)
    return None


def _rest_get(base_url, route, **params):
    resp = safe_get(
        base_url.rstrip('/') + '/',
        auth=_wp_auth(),
        params={'rest_route': route, **params},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _first_rest_match(base_url, content_type, slug):
    results = _rest_get(base_url, f'/wp/v2/{content_type}', slug=slug)
    return results[0] if results else None


def _front_page_id(base_url):
    """Find the static front page's post ID from the homepage's own public HTML.

    /wp/v2/settings (the REST source for page_on_front) requires manage_options,
    which an Editor-level Application Password never has — so this reads the
    page-id-{N} class WordPress's body_class() always emits for a singular page
    view, including a static front page. No auth needed.
    """
    try:
        resp = safe_get(base_url, timeout=10)
        resp.raise_for_status()
        match = re.search(r'page-id-(\d+)', resp.text)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def resolve_wp_object(url):
    """Find the WordPress content object for a crawled URL.

    Handles plain-permalink URLs (?p=123) directly, and falls back to the
    public posts/pages REST endpoints for pretty-permalink URLs. Home/front-page
    URLs resolve via the page-id-{N} class in the homepage's own HTML.
    Returns {'id': int, 'type': 'post'|'page', 'url': str}, or None.
    """
    target_url = map_to_target_url(url)
    parsed = urlparse(target_url)
    query = parse_qs(parsed.query)
    base_url = _base_url(target_url)

    if 'p' in query:
        return {'id': int(query['p'][0]), 'type': 'post', 'url': target_url}

    if 'page_id' in query:
        return {'id': int(query['page_id'][0]), 'type': 'page', 'url': target_url}

    path = parsed.path.strip('/')
    if not path:
        # An empty path with leftover query params (e.g. a calendar plugin's
        # ?method=ical&id=... export link) isn't the homepage — don't misresolve
        # it to the front page just because there's no path to match.
        if query:
            return None
        page_id = _front_page_id(base_url)
        if page_id:
            return {'id': page_id, 'type': 'page', 'url': target_url}
        return None

    slug = path.split('/')[-1]

    if not slug:
        return None

    for content_type, object_type in (('posts', 'post'), ('pages', 'page')):
        item = _first_rest_match(base_url, content_type, slug)
        if item:
            return {'id': item['id'], 'type': object_type, 'url': target_url}

    return None


def resolve_post_id(url):
    """Backward-compatible helper for old Agent 3 code."""
    wp_object = resolve_wp_object(url)
    return wp_object['id'] if wp_object else None


def probe_site(url):
    """Probe a site's public surface (no auth required) to approximate its plugin/theme stack.

    Used when no credentials/backup are available for a production site (e.g. Atlas) —
    reads /wp-json/ for active REST namespaces, and the homepage HTML for the
    generator tag, active theme, and plugin asset paths.

    Returns {'wp_version': str|None, 'namespaces': [...], 'theme': str|None, 'plugins': [...]}.
    """
    base = _base_url(map_to_target_url(url))

    namespaces = []
    try:
        resp = safe_get(f"{base}/wp-json/", timeout=10)
        resp.raise_for_status()
        namespaces = resp.json().get('namespaces', [])
    except Exception:
        pass

    plugins = set(NAMESPACE_PLUGIN_MAP[ns] for ns in namespaces if ns in NAMESPACE_PLUGIN_MAP)

    wp_version = None
    theme = None
    try:
        resp = safe_get(base, timeout=10)
        resp.raise_for_status()
        html = resp.text

        m = re.search(r'<meta name="generator" content="WordPress ([\d.]+)"', html)
        if m:
            wp_version = m.group(1)

        m = re.search(r'/wp-content/themes/([^/"\']+)/', html)
        if m:
            theme = m.group(1)

        plugins |= set(re.findall(r'/wp-content/plugins/([^/"\']+)/', html))
    except Exception:
        pass

    return {
        'wp_version': wp_version,
        'namespaces': namespaces,
        'theme': theme,
        'plugins': sorted(plugins),
    }


def apply_rankmath_meta(url, post_id, meta_dict):
    """Update RankMath SEO fields (title, description, canonical, etc.) for a post.

    RankMath fields aren't exposed on /wp/v2/posts, so this uses RankMath's own
    REST endpoint instead. Requires WP_USERNAME and WP_APP_PASSWORD in the
    environment (Application Password auth). Returns the parsed JSON response.
    """
    mapped_url = map_to_target_url(url)
    parsed = urlparse(mapped_url)
    api_url = f"{parsed.scheme}://{parsed.netloc}/?rest_route=/rankmath/v1/updateMeta"

    resp = safe_post(
        api_url,
        auth=_wp_auth(),
        json={'objectType': 'post', 'objectID': post_id, 'meta': meta_dict},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def apply_fix(url, post_id, object_type, field, new_value):
    """Update one field on a WordPress post or page via the core REST API.
    Requires WP_USERNAME and WP_APP_PASSWORD in the environment (Application Password auth).
    Returns the new value of the field as confirmed by WordPress.
    """
    parsed = urlparse(map_to_target_url(url))
    collection = 'pages' if object_type == 'page' else 'posts'
    api_url = f"{parsed.scheme}://{parsed.netloc}/wp-json/wp/v2/{collection}/{post_id}"

    resp = safe_post(
        api_url,
        auth=_wp_auth(),
        json={field: new_value},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()[field]


_WP_RESIZE_SUFFIX = re.compile(r'-\d+x\d+(?=\.\w+$)')


def _strip_wp_resize_suffix(filename):
    """Strip WordPress's auto-generated '-{width}x{height}' suffix from a filename
    (e.g. 'photo-1024x512.png' -> 'photo.png'). <img src> on a page almost always
    points to one of these generated variants, while the media library's own
    title/slug (what /wp/v2/media's search param matches against) is based on the
    original filename — searching on the un-stripped resized filename can return
    zero candidates even when the attachment genuinely exists."""
    return _WP_RESIZE_SUFFIX.sub('', filename)


def resolve_media_id(url, image_url):
    """Find the WordPress media library attachment ID for an image URL.

    <img src> on a page almost always references an auto-generated resized variant
    (e.g. photo-1024x512.png), not the original upload that WordPress registers as
    the attachment's own source_url (photo.png) — so this searches on the base
    filename (resize suffix stripped) and confirms the match by checking BOTH the
    attachment's source_url and every one of its registered media_details.sizes
    entries for an exact filename match against the real image_url.

    /wp/v2/media's search param matches title/caption/alt text too, not just the
    filename — trusting the first result risks writing alt text to the wrong
    attachment on a site with a large media library, so this still requires an
    exact filename match; it's just checked against every size WordPress
    generated instead of only the original upload. Returns None if no exact match
    is found (caller should defer rather than guess).
    """
    base_url = _base_url(map_to_target_url(url))
    filename = os.path.basename(urlparse(image_url).path)
    if not filename:
        return None
    search_term = _strip_wp_resize_suffix(filename)

    try:
        results = _rest_get(base_url, '/wp/v2/media', search=search_term, per_page=20)
    except Exception:
        return None

    for item in results:
        candidate_urls = [item.get('source_url', '')]
        sizes = item.get('media_details', {}).get('sizes', {})
        candidate_urls += [size.get('source_url', '') for size in sizes.values()]
        if any(os.path.basename(urlparse(u).path) == filename for u in candidate_urls if u):
            return item['id']
    return None


def apply_alt_text(url, media_id, alt_text):
    """Update a media attachment's alt_text field via the core REST API.
    Requires WP_USERNAME and WP_APP_PASSWORD in the environment (Application Password auth).
    Kept separate from apply_fix (which targets /posts and /pages) since media is a
    different endpoint with different field semantics — not just a different collection name.
    """
    parsed = urlparse(map_to_target_url(url))
    api_url = f"{parsed.scheme}://{parsed.netloc}/wp-json/wp/v2/media/{media_id}"

    resp = safe_post(
        api_url,
        auth=_wp_auth(),
        json={'alt_text': alt_text},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()['alt_text']


def get_raw_content(url, post_id, object_type):
    """Fetch a post/page's raw (unrendered) content — includes Gutenberg block
    comments/markup intact, unlike the public GET which only returns rendered HTML.
    Needed so a fix can locate and replace one specific block by exact text match
    rather than working blind against rendered output. Requires WP_USERNAME and
    WP_APP_PASSWORD (the edit context requires authentication).
    """
    parsed = urlparse(map_to_target_url(url))
    collection = 'pages' if object_type == 'page' else 'posts'
    api_url = f"{parsed.scheme}://{parsed.netloc}/wp-json/wp/v2/{collection}/{post_id}"

    resp = safe_get(api_url, auth=_wp_auth(), params={'context': 'edit'}, timeout=10)
    resp.raise_for_status()
    return resp.json()['content']['raw']


def get_plugin_status(base_url, slug):
    """Return 'active', 'inactive', or None (not installed) for a plugin slug.
    Requires an Administrator-level Application Password — Editor-level 403s on
    /wp/v2/plugins (install_plugins/activate_plugins are admin-only capabilities).
    """
    try:
        data = _rest_get(base_url, f'/wp/v2/plugins/{slug}')
        return data.get('status')
    except requests.exceptions.HTTPError:
        return None


def ensure_plugin_active(url, slug):
    """Install (from the wordpress.org plugin directory, by slug only — this
    endpoint does not accept zip uploads) and/or activate a plugin if it isn't
    already active. Requires WP_USERNAME/WP_APP_PASSWORD to be an Administrator
    — Editor-level Application Passwords can't reach /wp/v2/plugins at all.
    """
    base_url = _base_url(map_to_target_url(url))
    status = get_plugin_status(base_url, slug)

    if status is None:
        resp = safe_post(
            base_url.rstrip('/') + '/',
            auth=_wp_auth(),
            params={'rest_route': '/wp/v2/plugins'},
            json={'slug': slug},
            timeout=20,
        )
        resp.raise_for_status()
        status = resp.json().get('status')

    if status != 'active':
        resp = safe_put(
            base_url.rstrip('/') + '/',
            auth=_wp_auth(),
            params={'rest_route': f'/wp/v2/plugins/{slug}'},
            json={'status': 'active'},
            timeout=20,
        )
        resp.raise_for_status()

    return True


def get_default_redirect_group(base_url):
    """Return the id of the first redirect group (the Redirection plugin always
    creates a default 'Redirections' group on activation) — queried dynamically
    rather than assumed to be id 1, since that's not guaranteed per-site.
    """
    groups = _rest_get(base_url, '/redirection/v1/group')
    items = groups.get('items', groups) if isinstance(groups, dict) else groups
    return items[0]['id'] if items else None


def create_redirect_rule(url, source_path, target_url):
    """Create a single-hop 301 redirect rule via the Redirection plugin's own
    REST API (/redirection/v1/redirect — confirmed from the plugin's own REST
    controller source, johngodley/redirection api/api-redirect.php). Requires
    Redirection active and an Administrator-level Application Password.

    Field names below are derived from the plugin's source, not a live response
    — verify against the actual API once Redirection is installed and adjust if
    the live endpoint rejects this payload.
    """
    base_url = _base_url(map_to_target_url(url))
    group_id = get_default_redirect_group(base_url)

    resp = safe_post(
        base_url.rstrip('/') + '/',
        auth=_wp_auth(),
        params={'rest_route': '/redirection/v1/redirect'},
        json={
            'url': source_path,
            'match_type': 'url',
            'action_type': 'url',
            'action_data': {'url': target_url},
            'action_code': 301,
            'group_id': group_id,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
