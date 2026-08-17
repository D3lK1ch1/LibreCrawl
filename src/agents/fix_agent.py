"""
Agent 3 — Fix.
Triggered by: supervisor clicking "Run Agent 3" on an Azure DevOps ticket.
Input: Azure ticket ID. Output: SEO fix applied to the WordPress site via WP REST API,
ticket state updated, or deferral comment posted if the fix can't be automated.
Requires: RankMath plugin active on the target site for most FIX_MAP entries.
Does NOT verify that the fix worked — that is Agent 4's job.
"""
import os
import re
import html
import base64
import textwrap
import requests
from urllib.parse import quote, urlparse, urljoin
from bs4 import BeautifulSoup

from src.agents.wordpress import (
    resolve_wp_object, apply_rankmath_meta, apply_fix,
    probe_site, ensure_plugin_active, create_redirect_rule, NAMESPACE_PLUGIN_MAP,
    resolve_media_id, apply_alt_text, get_raw_content, map_to_target_url,
)
from src.agents.provider import call_with_tools
from src.agents.url_safety import safe_get
from src.agents.prompts import (
    build_plugin_decision_prompt, build_fix_title_prompt,
    build_fix_alt_text_prompt, build_fix_h1_prompt, build_fix_meta_description_prompt,
)

# Images processed per 'Images Without Alt Text' ticket — bounds cost/latency, since each
# image is one AI text call plus a WordPress write.
MAX_ALT_TEXT_IMAGES = 5

# Some sites (confirmed live on ci-dev.xyz) have hotlink protection that rejects a direct
# image request whose referrer isn't the site itself (Cloudflare error 1011) — a link straight
# to the image file fails when clicked from an Azure DevOps ticket, even though the image is
# fine on the live page. So the ticket links to the *page* instead (always works) and tells the
# reviewer how to actually check the image there, rather than offering a link that fails.
def _alt_text_view_instructions(page_url):
    safe_page_url = html.escape(page_url)
    return (f'Open <a href="{safe_page_url}">the page</a> and use View Page Source or Inspect '
            f'Element — search for each filename below to see its current alt attribute. '
            f'(Direct image links are blocked by this site\'s hotlink protection.)')


def _confirm_rendered(url, expected):
    """Re-fetch url and check whether each expected {meta-name: value} pair actually
    appears in the rendered HTML. Returns the subset of expected keys NOT found —
    empty dict means everything rendered. The '__title__' key checks the <title>
    element instead of a <meta> tag."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        return {'__fetch_error__': str(e)}

    missing = {}
    for name, value in expected.items():
        if name == '__title__':
            # startswith, not ==: the theme appends its own " | Site Name" suffix
            # to every rendered <title>, which Agent 3 never writes and can't remove.
            rendered = soup.title.string.strip() if soup.title and soup.title.string else None
            ok = rendered is not None and rendered.startswith(value)
        else:
            tag = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
            rendered = tag.get('content', '') if tag else None
            ok = rendered == value
        if not ok:
            missing[name] = f'expected {value!r}, found {rendered!r}'
    return missing


def _confirm_canonical_rendered(url, expected_href):
    """Re-fetch url and check whether <link rel="canonical"> matches expected_href.
    Returns None if it matches, or a reason string if it's missing/mismatched.
    Confirmed live on ci-dev.xyz: RankMath can persist a canonical value (visible in
    its own Advanced tab) without ever outputting the <link> tag on the page — that's
    a RankMath output/config issue, not a write failure, so the reason says so instead
    of just 'not found'."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"[Agent 3] Could not re-fetch {url} to verify render: {e}")
        return 'could not re-fetch page to verify — see server logs'

    tag = soup.find('link', rel='canonical')
    rendered = tag.get('href') if tag else None
    if rendered == expected_href:
        return None
    if rendered is None:
        return ('canonical value was written but no &lt;link rel="canonical"&gt; tag is rendered on the '
                 'page — looks like a RankMath output/configuration issue, not a write failure')
    return f'canonical value was written but rendered as {rendered!r}, expected {expected_href!r}'


def _fetch_current_meta_description(url):
    """Read whatever <meta name="description"> currently renders on the live page, before it
    gets overwritten — same 'read the live rendered state' approach fix_title_core already uses
    for its before/after, not a RankMath-storage read (RankMath's REST routes have no read-back
    endpoint — confirmed live, only updateMeta/updateMetaBulk exist). Returns None if there's
    no current description tag at all, which is the accurate 'before' for a Missing ticket."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return None
    tag = soup.find('meta', attrs={'name': 'description'})
    return tag.get('content') if tag else None


def _confirm_meta_description_rendered(url, expected_description):
    """Re-fetch url and check whether <meta name="description"> matches expected_description.
    Returns None if it matches, or a reason string if it's missing/mismatched. Same
    RankMath-output-isn't-guaranteed risk as canonical (_confirm_canonical_rendered): confirmed
    live that this site's RankMath/theme setup can fail to render one Head-module field
    (canonical) while OpenGraph/Social tags render fine on the same page — description hasn't
    been assumed safe from the same failure, so this checks it explicitly instead."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"[Agent 3] Could not re-fetch {url} to verify render: {e}")
        return 'could not re-fetch page to verify — see server logs'

    tag = soup.find('meta', attrs={'name': 'description'})
    rendered = tag.get('content') if tag else None
    if rendered == expected_description:
        return None
    if rendered is None:
        return ('description value was written but no &lt;meta name="description"&gt; tag is rendered on '
                 'the page — looks like a RankMath output/configuration issue, not a write failure')
    return f'description value was written but rendered as {rendered!r}, expected {expected_description!r}'


def _confirm_h1_rendered(url):
    """Re-fetch url and check whether an <h1> now exists anywhere on the rendered page.
    Returns None if it does, or a reason string if the write didn't produce one."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"[Agent 3] Could not re-fetch {url} to verify render: {e}")
        return 'could not re-fetch page to verify — see server logs'
    return None if soup.find('h1') else 'write succeeded but no &lt;h1&gt; is rendered on the page'


def _admin_edit_link(site_url, object_id):
    """Build a WP admin edit-screen URL for a post, page, or media attachment ID — all
    three use the same post.php?post={id}&action=edit editor route. Included in every
    fix caveat so a human reviewing the ticket has an exact place to look, not just a
    claim that something changed."""
    parsed = urlparse(site_url)
    return f"{parsed.scheme}://{parsed.netloc}/wp-admin/post.php?post={object_id}&action=edit"


def _wp_admin_check_link(site_url, object_type, object_id):
    """HTML anchor to the WP admin edit screen, with link text naming what it is. Azure
    DevOps's comments API renders HTML in the ticket discussion, so this shows up as an
    actual clickable link there — not a raw URL the reviewer has to copy-paste."""
    return f'<a href="{_admin_edit_link(site_url, object_id)}">{object_type} ID {object_id} in wp-admin</a>'


def _fetch_page_context(url):
    """Re-fetch a page and pull title/H1/opening body text for LLM prompts."""
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else ''
        h1 = soup.find('h1')
        h1_text = h1.get_text(strip=True) if h1 else ''
        body = soup.find('body')
        body_text = ' '.join(body.get_text(' ', strip=True).split())[:500] if body else ''
        return {'title': title, 'h1': h1_text, 'body_excerpt': body_text}
    except Exception:
        return {'title': '', 'h1': '', 'body_excerpt': ''}


def _fetch_images_without_alt(url):
    """Re-fetch the live page and find <img> tags with no alt attribute at all, along with
    surrounding DOM/text context (figcaption, nearest heading, nearby paragraph text, title
    attribute) so alt text can be generated from that context instead of a vision call on the
    image pixels — see fix_images_alt_text(). The ticket's details field is only a count
    string ("3 of 10 images lack alt text") — no per-image URLs travel with it — so both the
    image list and its context have to be re-derived here, same re-fetch-at-fix-time pattern
    as _fetch_page_context.
    """
    try:
        resp = safe_get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
    except Exception:
        return []

    images = []
    for img in soup.find_all('img'):
        # Matches issue_detector.py's own definition of this issue (`not img.get('alt')`) —
        # alt="" (present but empty, Gutenberg's default for a never-filled-in field) counts
        # as missing too, not just an absent attribute. A prior version of this check used
        # `is not None`, which treated alt="" as "fine" and silently found nothing to fix.
        if img.get('alt'):
            continue
        src = img.get('src')
        if not src:
            continue
        abs_src = urljoin(url, src)

        figure = img.find_parent('figure')
        figcaption = figure.find('figcaption') if figure else None
        heading = img.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        prev_p = img.find_previous('p')
        next_p = img.find_next('p')

        images.append({
            'src': abs_src,
            'filename': os.path.basename(urlparse(abs_src).path),
            'title_attr': (img.get('title') or '').strip(),
            'figcaption': figcaption.get_text(' ', strip=True) if figcaption else '',
            'nearby_heading': heading.get_text(' ', strip=True) if heading else '',
            'surrounding_text': ' '.join(t.get_text(' ', strip=True) for t in (prev_p, next_p) if t).strip(),
        })
    return images


def decide_required_plugin(issue, active_namespaces):
    """Ask the LLM which plugin (if any) is needed to fix this issue, constrained to
    NAMESPACE_PLUGIN_MAP's known candidates (rankmath/v1, redirection/v1) — never an
    open-ended choice from the whole plugin directory. Any answer outside the allowed
    set is treated as 'none', so a malformed/hallucinated reply fails safe to Defer
    rather than triggering an install.
    """
    candidates = {ns: slug for ns, slug in NAMESPACE_PLUGIN_MAP.items() if ns in ('rankmath/v1', 'redirection/v1')}
    prompt = build_plugin_decision_prompt(issue, active_namespaces, list(candidates.values()))
    _, text, _, _ = call_with_tools([{'role': 'user', 'content': prompt}], [], agent="agent3", label=f"Plugin check — {issue}")
    answer = text.strip().strip("'\"")
    return answer if answer in candidates.values() else 'none'


def _trace_redirect_chain(url, max_hops=10):
    """Walk a redirect chain one hop at a time with allow_redirects=False. The
    crawler itself fetches with allow_redirects=True (the default), which collapses
    chains transparently and never records individual hops — crawled_urls.redirects
    is always an empty list — so this re-fetches live at fix-time instead.
    Returns [(url, status_code), ...] ending with the final non-redirect response.
    """
    chain, current = [], url
    for _ in range(max_hops):
        resp = safe_get(current, allow_redirects=False, timeout=10)
        chain.append((current, resp.status_code))
        if resp.status_code in (301, 302, 303, 307, 308) and 'Location' in resp.headers:
            current = urljoin(current, resp.headers['Location'])
        else:
            break
    return chain


def _redirect_confidence_check(url):
    """notes/agent3-fix-or-defer-spec-2026-06-15.md §7's rule: only safe to collapse
    if every hop is a same-domain 301 ending in a single final 200. Returns the final
    URL if safe to collapse, or None if it should stay Defer (multi-hop judgment call,
    cross-domain, or doesn't resolve to a working page).
    """
    chain = _trace_redirect_chain(url)
    if len(chain) < 2:
        return None
    *hops, (final_url, final_status) = chain
    base_domain = urlparse(url).netloc
    if final_status != 200:
        return None
    if any(status != 301 or urlparse(hop_url).netloc != base_domain for hop_url, status in hops):
        return None
    return final_url


_TITLE_SEPARATORS = (' | ', ' — ', ' – ', ' - ')


def _detect_title_suffix(current_title):
    """Find the theme's appended '<sep> Site Name' suffix on a rendered title, if any.
    Checks a handful of common separators rather than assuming one — a real run on
    ci-dev.xyz rendered a title 76 chars long against a 60-char budget because this
    used to hardcode ' | ' only; that theme's actual separator is ' - ', so the
    30-char suffix went undetected, the budget came back as a full 60 with no room
    reserved for it, and the write blew the target by exactly the suffix's width.
    """
    for sep in _TITLE_SEPARATORS:
        idx = current_title.rfind(sep)
        if idx != -1:
            return current_title[idx:]
    return ''


def fix_title_core(ticket):
    """LLM-generated title, sized to the ticket's specific complaint. Written via
    core REST's native `title` field — no RankMath/plugin dependency.

    The rendered <title> tag always has the theme's own "<sep> Site Name" suffix
    appended after whatever we write here (confirmed live on ci-dev.xyz — the
    theme does this for every page, Agent 3 never touches it). The 60-char
    target is for the *rendered* title, so the budget for our own text has to
    leave room for that suffix, not aim for 60 raw characters. Suffix is read
    from the page's current title rather than hardcoded, so this still works
    on pages/themes with a different or no suffix — see _detect_title_suffix().

    If the AI still overshoots its budget, shorten() cuts at a word boundary
    instead of a blind character slice — a mid-word cut previously produced
    grammatically broken titles (e.g. "Cultural Infusion" -> "Cultural Infus").
    """
    ctx = _fetch_page_context(ticket['url'])
    current_title = ctx['title']
    suffix = _detect_title_suffix(current_title)
    budget = max(60 - len(suffix), 20)
    length = f"between 30 and {budget} characters" if 'Short' in ticket['issue'] else f"at most {budget} characters"
    prompt = build_fix_title_prompt(length, current_title, ctx['h1'], ctx['body_excerpt'])
    _, text, _, _ = call_with_tools([{'role': 'user', 'content': prompt}], [], agent="agent3", label=ticket['issue'])
    title = text.strip()
    if len(title) > budget:
        title = textwrap.shorten(title, width=budget, placeholder='', break_on_hyphens=False)
    return title, current_title


def fix_images_alt_text(ticket):
    """Fix for 'Images Without Alt Text' — alt text generated from page/DOM context around
    each image (filename, figcaption, nearest heading, surrounding paragraph text) via a
    text-only LLM call, not a vision call on the image pixels — cuts token cost per image.
    Written directly to each affected image's WordPress media attachment. Capped at
    MAX_ALT_TEXT_IMAGES per ticket (each image costs one AI text call plus one WP write).
    Returns a full run_fix-style
    result dict covering all images processed, rather than the single RankMath meta dict
    other FIX_MAP entries return — alt text isn't a RankMath field, so this is dispatched as
    its own special-cased branch in run_fix, not through apply_rankmath_meta.
    """
    url = ticket['url']
    issue = ticket['issue']
    ctx = _fetch_page_context(url)
    images = _fetch_images_without_alt(url)[:MAX_ALT_TEXT_IMAGES]

    if not images:
        return {'status': 'skipped',
                'reason': ("Agent 3 re-fetched the live page and scanned every &lt;img&gt; tag for missing "
                           "or empty alt text, but found none remaining — the page may have changed "
                           "since this issue was crawled, or the affected images are no longer present."),
                'issue': issue, 'url': url}

    results = []
    for img in images:
        image_url = img['src']
        prompt = build_fix_alt_text_prompt(ctx['title'], img)
        try:
            _, text, _, _ = call_with_tools([{'role': 'user', 'content': prompt}], [],
                                             agent="agent3", label=f"Alt text — {image_url}")
        except Exception as e:
            print(f"[Agent 3] Could not generate alt text for {image_url}: {e}")
            results.append({'image_url': image_url, 'status': 'error', 'reason': 'the AI call failed — see server logs'})
            continue

        if not text.strip():
            reason = 'the AI call returned no usable text (no provider configured, or the model returned an empty response)'
            print(f"[Agent 3] Could not generate alt text for {image_url}: {reason}")
            results.append({'image_url': image_url, 'status': 'error', 'reason': reason})
            continue

        alt_text = '' if text.strip().upper() == 'DECORATIVE' else text.strip()

        media_id = resolve_media_id(url, image_url)
        if media_id is None:
            print(f"[Agent 3] Could not resolve media attachment for {image_url} — skipping.")
            results.append({'image_url': image_url, 'status': 'error',
                             'reason': (f"generated {alt_text!r}, but could not find a matching media "
                                        f"library attachment for this image to write it to")})
            continue

        try:
            apply_alt_text(url, media_id, alt_text)
        except Exception:
            print(f"[Agent 3] Could not apply alt text for {image_url} — WordPress write failed.")
            results.append({'image_url': image_url, 'status': 'error',
                             'reason': f"generated {alt_text!r}, but the write to WordPress failed"})
            continue

        print(f"[Agent 3] Alt text applied for {image_url}: {alt_text!r}")
        results.append({'image_url': image_url, 'status': 'fixed', 'alt_text': alt_text, 'media_id': media_id})

    fixed_count = sum(1 for r in results if r['status'] == 'fixed')
    failures = [r for r in results if r['status'] == 'error']

    # HTML list, not a semicolon-joined sentence — Azure DevOps comments render HTML, so this
    # shows up as an actual bulleted list, not a wall of text the reviewer has to parse. Rows
    # name the image's full path (not just the bare filename) to search for in Page Source —
    # a bare filename like "1.jpg" is too generic to Ctrl+F reliably (collides with resized
    # variants, other uploads, or unrelated "1.jpg" text elsewhere on the page); the full path
    # is guaranteed to match the exact <img> tag's src attribute, wherever it appears verbatim.
    def _fixed_row(r):
        path = html.escape(urlparse(r['image_url']).path)
        edit_url = html.escape(_admin_edit_link(map_to_target_url(url), r['media_id']))
        alt_text = html.escape(r['alt_text'])
        return (f'<li>"{path}" — generated: "{alt_text}" — '
                f'<a href="{edit_url}">Edit in wp-admin</a></li>')

    def _failed_row(r):
        path = html.escape(urlparse(r['image_url']).path)
        reason = html.escape(r['reason'])
        return f'<li>"{path}" — {reason}</li>'

    view_instructions = _alt_text_view_instructions(url)

    if fixed_count:
        fixed_list = ''.join(_fixed_row(r) for r in results if r['status'] == 'fixed')
        caveat = (
            f"AI-generated alt text applied to {fixed_count}/{len(images)} image(s) "
            f"(capped at {MAX_ALT_TEXT_IMAGES} per ticket). {view_instructions}"
            f"<ul>{fixed_list}</ul>"
        )
        if failures:
            failed_list = ''.join(_failed_row(r) for r in failures)
            caveat += f"{len(failures)} image(s) could not be processed:<ul>{failed_list}</ul>"
        return {'status': 'fixed', 'issue': issue, 'url': url, 'images': results, 'caveat': caveat}

    failed_list = ''.join(_failed_row(r) for r in failures)
    return {
        'status': 'error',
        'issue': issue,
        'url': url,
        'images': results,
        'reason': (f"Agent 3 found {len(images)} image(s) without alt text and attempted to generate "
                   f"and apply alt text for each, but all failed. {view_instructions}"
                   f"<ul>{failed_list}</ul>"),
    }


def _find_heading_block(raw_content, text):
    """Find an existing Gutenberg heading block (any level h1-h6) whose inner text is
    exactly `text`. Returns the single re.Match if found exactly once, else None —
    zero or multiple matches are both too ambiguous to touch safely."""
    escaped = re.escape(text.strip())
    pattern = re.compile(
        r'<!-- wp:heading(?: (\{[^}]*\}))? -->\s*<h([1-6])([^>]*)>\s*' + escaped + r'\s*</h\2>\s*<!-- /wp:heading -->'
    )
    matches = list(pattern.finditer(raw_content))
    return matches[0] if len(matches) == 1 else None


def _find_paragraph_block(raw_content, text):
    """Find an existing Gutenberg paragraph block whose inner text is exactly `text`.
    Returns the single re.Match if found exactly once, else None."""
    escaped = re.escape(text.strip())
    pattern = re.compile(r'<!-- wp:paragraph[^>]*-->\s*<p[^>]*>\s*' + escaped + r'\s*</p>\s*<!-- /wp:paragraph -->')
    matches = list(pattern.finditer(raw_content))
    return matches[0] if len(matches) == 1 else None


def fix_missing_h1(ticket):
    """Fix for 'Missing H1 Tag' — AI scans the post's raw block content for an existing
    block whose text reads like it was meant to be the page's main heading, then promotes
    just that one block to <h1> via a precise, exact-match substitution. Never a blind
    full-content rewrite: if the candidate text doesn't match raw content exactly once
    (ambiguous, or the page uses a page-builder's own non-block markup — e.g. Cornerstone
    — that won't match Gutenberg block patterns at all), this defers instead of guessing.
    Confirmed live: on ci-dev.xyz the common case is the post title already rendered as
    an <h2>, which this promotes to <h1> rather than writing new heading text.
    """
    url = ticket['url']
    issue = ticket['issue']

    try:
        wp_object = resolve_wp_object(url)
    except Exception:
        wp_object = None
    if wp_object is None:
        return {'status': 'skipped', 'reason': _defer_missing_h1(ticket), 'issue': issue, 'url': url}

    print("[Agent 3] Fetching raw content to locate a heading candidate...")
    try:
        raw_content = get_raw_content(url, wp_object['id'], wp_object['type'])
    except Exception:
        print("[Agent 3] Could not fetch raw content — request to WordPress failed.")
        return {'status': 'error',
                'reason': (f"Agent 3 resolved this to {wp_object['type']} ID {wp_object['id']} and "
                           f"attempted to fetch its raw content to look for a heading candidate, but "
                           f"the request to WordPress failed."),
                'issue': issue, 'url': url}

    ctx = _fetch_page_context(url)
    prompt = build_fix_h1_prompt(raw_content[:8000], ctx['title'])
    _, text, _, _ = call_with_tools([{'role': 'user', 'content': prompt}], [], agent="agent3", label=issue)
    candidate = text.strip().strip("'\"")
    # Real page content, verbatim — could contain '&'/'<'/'>' that would otherwise break the
    # HTML the ticket comment renders as. The literal `candidate` (unescaped) is still what
    # gets written into new_block below — that goes into the actual WordPress content field,
    # not a comment, and must stay as real markup.
    safe_candidate = html.escape(candidate)

    if not candidate or candidate.upper() == 'NONE':
        return {
            'status': 'skipped',
            'reason': (f"Agent 3 resolved this to {wp_object['type']} ID {wp_object['id']} and scanned "
                       f"its content for text that could serve as the page's main heading, but found no "
                       f"clear candidate — needs a human to write/approve one. Check "
                       f"{_wp_admin_check_link(wp_object['url'], wp_object['type'], wp_object['id'])}"),
            'issue': issue, 'url': url,
        }

    heading_match = _find_heading_block(raw_content, candidate)
    paragraph_match = _find_paragraph_block(raw_content, candidate)

    if heading_match:
        old_level = heading_match.group(2)
        extra_attrs = heading_match.group(3) or ''
        new_block = f'<!-- wp:heading {{"level":1}} -->\n<h1{extra_attrs}>{candidate}</h1>\n<!-- /wp:heading -->'
        match = heading_match
        promoted_from = f'existing <h{old_level}> heading'
    elif paragraph_match:
        new_block = f'<!-- wp:heading {{"level":1}} -->\n<h1>{candidate}</h1>\n<!-- /wp:heading -->'
        match = paragraph_match
        promoted_from = 'existing paragraph text'
    else:
        return {
            'status': 'skipped',
            'reason': (f"AI suggested {safe_candidate!r} as the heading, but it doesn't match this page's "
                       f"raw content exactly once — too ambiguous to promote safely; needs a human to "
                       f"write/approve the heading. Check {wp_object['type']} ID {wp_object['id']} in "
                       f"wp-admin: {_admin_edit_link(wp_object['url'], wp_object['id'])}"),
            'issue': issue, 'url': url,
        }

    new_content = raw_content[:match.start()] + new_block + raw_content[match.end():]

    print(f"[Agent 3] Applying fix: promoting {promoted_from} to <h1> on {wp_object['type']} {wp_object['id']}...")
    try:
        apply_fix(url, wp_object['id'], wp_object['type'], 'content', new_content)
    except Exception:
        print("[Agent 3] Could not apply fix — WordPress write failed.")
        return {'status': 'error',
                'reason': (f"Agent 3 identified {promoted_from} ({safe_candidate!r}) to promote to &lt;h1&gt; and "
                           f"attempted to write the updated content, but the write to WordPress failed. Check "
                           f"{_wp_admin_check_link(wp_object['url'], wp_object['type'], wp_object['id'])}"),
                'issue': issue, 'url': url,
                'object_id': wp_object['id'], 'object_type': wp_object['type']}

    render_issue = _confirm_h1_rendered(url)
    if render_issue:
        print(f"[Agent 3] Write succeeded but H1 did not render: {render_issue}")
        return {'status': 'error',
                'reason': (f"Agent 3 promoted {promoted_from} ({safe_candidate!r}) to &lt;h1&gt; and the write to "
                           f"WordPress succeeded, but re-checking the live page found: {render_issue}. Check "
                           f"{_wp_admin_check_link(wp_object['url'], wp_object['type'], wp_object['id'])}"),
                'issue': issue, 'url': url,
                'object_id': wp_object['id'], 'object_type': wp_object['type']}

    print(f"[Agent 3] Fix applied and confirmed rendered. Promoted {promoted_from} to H1: {candidate!r}")
    return {
        'status': 'fixed',
        'issue': issue,
        'url': url,
        'object_id': wp_object['id'],
        'object_type': wp_object['type'],
        'meta': {'h1': candidate},
        'caveat': (f"Promoted {promoted_from} to &lt;h1&gt; based on AI content analysis: {safe_candidate!r}. "
                   f"Please double-check this heading is appropriate — check "
                   f"{_wp_admin_check_link(wp_object['url'], wp_object['type'], wp_object['id'])}"),
    }


def _generate_description(url, length):
    """LLM-generated description grounded in the page's actual title/h1/body content —
    shared by meta description, OG description, and Twitter description, which are the
    same kind of text at different length budgets. Replaces what used to be a hardcoded
    '[Agent 3] Description for: <url>' placeholder written with no LLM call at all —
    that placeholder shipped as the literal live meta description/OG/Twitter content on
    every ticket of these types, visible in search results and social previews."""
    ctx = _fetch_page_context(url)
    prompt = build_fix_meta_description_prompt(length, ctx['title'], ctx['h1'], ctx['body_excerpt'])
    _, text, _, _ = call_with_tools([{'role': 'user', 'content': prompt}], [], agent="agent3", label=f"Description — {url}")
    return text.strip()


def fix_meta_description(ticket):
    """Fix for 'Missing/Too Long/Too Short Meta Description' — AI-generated description
    grounded in the page's actual content, sets RankMath's description field."""
    old_description = _fetch_current_meta_description(ticket['url']) or '(none)'
    description = _generate_description(ticket['url'], 'between 120 and 160 characters')
    if len(description) > 160:
        description = textwrap.shorten(description, width=160, placeholder='', break_on_hyphens=False)
    caveat = (f"AI-generated meta description, based on this page's actual content. Please verify "
              f"it reads correctly.<ul><li>Before: {html.escape(old_description)!r}</li>"
              f"<li>After: {html.escape(description)!r}</li></ul>")
    return {'rank_math_description': description}, caveat


def fix_canonical_url(ticket):
    """Fix for 'Missing Canonical URL' — points RankMath's canonical field at the page's
    own URL. No LLM needed: self-referencing is the correct default for a *missing*
    canonical. Left un-generated caveat-wise since there's no guess involved here."""
    return {'rank_math_canonical_url': ticket['url']}, None


def fix_og_tags(ticket):
    """Fix for 'Missing OpenGraph Tags' — OG title reuses the page's real, already-correct
    title (no AI needed for that); OG description is AI-generated from actual page content."""
    ctx = _fetch_page_context(ticket['url'])
    description = _generate_description(ticket['url'], 'between 150 and 200 characters')
    if len(description) > 200:
        description = textwrap.shorten(description, width=200, placeholder='', break_on_hyphens=False)
    caveat = ("OpenGraph title reused from this page's actual title; description is "
              "AI-generated from page content. Please verify both.")
    return {
        'rank_math_facebook_title': (ctx['title'] or f"Agent 3 fix: {ticket['url']}")[:60],
        'rank_math_facebook_description': description,
    }, caveat


def fix_twitter_tags(ticket):
    """Fix for 'Missing Twitter Card Tags' — same approach as fix_og_tags: real title
    reused as-is, description AI-generated from actual page content."""
    ctx = _fetch_page_context(ticket['url'])
    description = _generate_description(ticket['url'], 'between 150 and 200 characters')
    if len(description) > 200:
        description = textwrap.shorten(description, width=200, placeholder='', break_on_hyphens=False)
    caveat = ("Twitter Card title reused from this page's actual title; description is "
              "AI-generated from page content. Please verify both.")
    return {
        'rank_math_twitter_title': (ctx['title'] or f"Agent 3 fix: {ticket['url']}")[:60],
        'rank_math_twitter_description': description,
    }, caveat


# Maps Azure ticket issue names (src/core/issue_detector.py) to fix functions.
# Each fix function takes the ticket dict and returns a dict of RankMath meta
# keys -> new values, passed to apply_rankmath_meta(). Anything not listed
# here is skipped — matches the documented Agent 3 fix map. Issues whose fix
# doesn't produce a RankMath meta dict are special-cased directly in run_fix()
# instead of living here: title issues (core REST write, see fix_title_core()),
# redirect issues (Redirection plugin), 'Images Without Alt Text' (media
# library write, see fix_images_alt_text()), and 'Missing H1 Tag' (post content
# write, see fix_missing_h1()).
FIX_MAP = {
    'Missing Meta Description': fix_meta_description,
    'Meta Description Too Long': fix_meta_description,
    'Meta Description Too Short': fix_meta_description,
    'Missing Canonical URL': fix_canonical_url,
    'Canonical URL Different': fix_canonical_url,
    'Missing OpenGraph Tags': fix_og_tags,
    'Missing Twitter Card Tags': fix_twitter_tags,
}

# Same three issue names run_fix()'s title branch matches on below — named here instead
# of duplicated as a second inline tuple, so there's one literal list to keep in sync.
TITLE_ISSUES = {'Missing Title Tag', 'Title Too Long', 'Title Too Short'}

# Every issue name whose fix requires resolving the URL to an editable WordPress post/page
# object first (run_fix() gates all of these behind resolve_wp_object() before generating
# any fix — see the title branch, fix_missing_h1(), and the FIX_MAP dispatch below). This
# is the complete list and nothing else: DEFER_REASONS/DEFER_PATTERNS issue types (Slow
# Response Time, Broken Image, Noindex, 404s, redirects, etc.) never touch WordPress
# resolution at all and must never be checked against this set — they're deferred for
# reasons unrelated to whether a WP object exists, and would be silently and wrongly
# dropped if this were ever applied to them (e.g. a real Broken Image issue on a page
# that happens to be a category archive is still a real, human-actionable issue).
# Consumed by ticket_review_agent.py's Agent 2 pre-check — grows automatically if FIX_MAP
# grows, since it's derived from FIX_MAP.keys() rather than a second hand-maintained list.
WP_OBJECT_REQUIRED_ISSUES = TITLE_ISSUES | {'Missing H1 Tag'} | set(FIX_MAP.keys())


# Known REST namespaces for common WordPress image-compression plugins — checked against
# probe_site()'s raw namespace list when deferring a page-size issue, so the ticket comment
# can name what was actually looked for rather than assuming nothing is installed.
IMAGE_COMPRESSION_NAMESPACES = ('shortpixel/v1', 'wp-smushit/v1', 'imagify/v1', 'ewww-image-optimizer/v1')


def _extract_broken_image_url(details):
    """Pull the image URL out of a 'Broken Image' ticket's details string
    (issue_detector.py's _check_broken_image_issues formats these as
    'Image does not respond: <url>' or 'Image returned <status>: <url>')."""
    match = re.search(r'(?:Image returned \d+|Image does not respond):\s*(.+)', details or '')
    return match.group(1).strip() if match else ''


def _defer_broken_image(ticket):
    """Re-check the broken image right now rather than repeating the crawl-time finding —
    it may have been fixed (or gone further stale) since the original crawl."""
    img_url = _extract_broken_image_url(ticket.get('details', ''))
    if not img_url:
        return "Referenced image doesn't exist — needs a human to source/select a replacement asset."
    try:
        resp = requests.head(img_url, allow_redirects=True, timeout=10)
        status = resp.status_code
    except Exception as e:
        return (f"Re-checked at defer-time: {img_url} still doesn't respond ({e}) — "
                f"needs a human to source/select a replacement asset.")
    if status < 400:
        return (f"Re-checked at defer-time: {img_url} now returns {status} — may have been fixed "
                f"since the original crawl. Verify and re-run the crawl if so; left deferred since "
                f"Agent 3 doesn't re-validate crawl results on its own.")
    return (f"Re-checked at defer-time: {img_url} → {status}. Still broken — needs a human to "
            f"source/select a replacement asset.")


def _defer_page_size(ticket):
    """Check which image-compression plugins are actually active on the site before deferring,
    instead of assuming none is present."""
    details = ticket.get('details', '')
    try:
        namespaces = probe_site(ticket['url'])['namespaces']
    except Exception:
        namespaces = []
    found = [ns for ns in IMAGE_COMPRESSION_NAMESPACES if ns in namespaces]
    if found:
        return (f"{details} — an image-compression plugin ({', '.join(found)}) is active on this "
                f"site, but Agent 3 doesn't assume its settings are safe to change automatically; "
                f"needs a human to review its compression settings for this page's images.")
    return (f"{details} — checked active plugins on this site for image compression "
            f"({', '.join(IMAGE_COMPRESSION_NAMESPACES)}) and found none active; install and "
            f"configure one, then re-run Agent 3.")


def _defer_missing_h1(ticket):
    """Distinguish an archive/listing page — whose H1 (if any) comes from the theme
    template, not editable post content — from a real single post/page, since only the
    latter is even theoretically fixable by writing to WordPress content. Confirmed live:
    ci-dev.xyz/blog/ has no <h1> anywhere in its rendered source at all, and resolve_wp_object
    can't match it to a post/page — a content write there wouldn't have anywhere to land."""
    try:
        wp_object = resolve_wp_object(ticket['url'])
    except Exception:
        wp_object = None
    if wp_object is None:
        return ("Could not resolve a WordPress post/page object for this URL — likely a "
                 "theme-generated archive/listing page (e.g. the blog index) rather than "
                 "editable content, so any H1 here comes from the theme template, not post "
                 "content. Needs a human/developer to add a heading in the theme template "
                 "directly — not something Agent 3 can fix by writing to a post.")
    return "Content judgment call — needs a human to write/approve the heading."


# Known REST namespaces for common WordPress image-compression plugins — checked against
# probe_site()'s raw namespace list when deferring a page-size issue, so the ticket comment
# can name what was actually looked for rather than assuming nothing is installed.
IMAGE_COMPRESSION_NAMESPACES = ('shortpixel/v1', 'wp-smushit/v1', 'imagify/v1', 'ewww-image-optimizer/v1')


def _extract_broken_image_url(details):
    """Pull the image URL out of a 'Broken Image' ticket's details string
    (issue_detector.py's _check_broken_image_issues formats these as
    'Image does not respond: <url>' or 'Image returned <status>: <url>')."""
    match = re.search(r'(?:Image returned \d+|Image does not respond):\s*(.+)', details or '')
    return match.group(1).strip() if match else ''


def _defer_broken_image(ticket):
    """Re-check the broken image right now rather than repeating the crawl-time finding —
    it may have been fixed (or gone further stale) since the original crawl."""
    img_url = _extract_broken_image_url(ticket.get('details', ''))
    if not img_url:
        return "Referenced image doesn't exist — needs a human to source/select a replacement asset."
    try:
        resp = requests.head(img_url, allow_redirects=True, timeout=10)
        status = resp.status_code
    except Exception as e:
        return (f"Re-checked at defer-time: {img_url} still doesn't respond ({e}) — "
                f"needs a human to source/select a replacement asset.")
    if status < 400:
        return (f"Re-checked at defer-time: {img_url} now returns {status} — may have been fixed "
                f"since the original crawl. Verify and re-run the crawl if so; left deferred since "
                f"Agent 3 doesn't re-validate crawl results on its own.")
    return (f"Re-checked at defer-time: {img_url} → {status}. Still broken — needs a human to "
            f"source/select a replacement asset.")


def _defer_page_size(ticket):
    """Check which image-compression plugins are actually active on the site before deferring,
    instead of assuming none is present."""
    details = ticket.get('details', '')
    try:
        namespaces = probe_site(ticket['url'])['namespaces']
    except Exception:
        namespaces = []
    found = [ns for ns in IMAGE_COMPRESSION_NAMESPACES if ns in namespaces]
    if found:
        return (f"{details} — an image-compression plugin ({', '.join(found)}) is active on this "
                f"site, but Agent 3 doesn't assume its settings are safe to change automatically; "
                f"needs a human to review its compression settings for this page's images.")
    return (f"{details} — checked active plugins on this site for image compression "
            f"({', '.join(IMAGE_COMPRESSION_NAMESPACES)}) and found none active; install and "
            f"configure one, then re-run Agent 3.")


def _defer_missing_h1(ticket):
    """Distinguish an archive/listing page — whose H1 (if any) comes from the theme
    template, not editable post content — from a real single post/page, since only the
    latter is even theoretically fixable by writing to WordPress content. Confirmed live:
    ci-dev.xyz/blog/ has no <h1> anywhere in its rendered source at all, and resolve_wp_object
    can't match it to a post/page — a content write there wouldn't have anywhere to land."""
    try:
        wp_object = resolve_wp_object(ticket['url'])
    except Exception:
        wp_object = None
    if wp_object is None:
        return ("Could not resolve a WordPress post/page object for this URL — likely a "
                 "theme-generated archive/listing page (e.g. the blog index) rather than "
                 "editable content, so any H1 here comes from the theme template, not post "
                 "content. Needs a human/developer to add a heading in the theme template "
                 "directly — not something Agent 3 can fix by writing to a post.")
    return "Content judgment call — needs a human to write/approve the heading."


# Issue types Agent 3 cannot fix regardless of access level (notes/agent3-fix-or-defer-spec-2026-06-15.md,
# Section 5). Exact-match issue names -> human-readable reason for the deferred-ticket comment, OR a
# callable taking the full ticket dict for reasons that need to interpolate ticket details or re-check
# something live (matches _confirm_canonical_rendered's pattern: what was attempted, what was found,
# what to do next — not a category label restated as a sentence).
DEFER_REASONS = {
    # 3a — outside any access method's reach (infra/network, before WordPress is even reached)
    'DNS Not Found': "DNS-level error — outside WordPress/Agent 3 scope, needs infra investigation.",
    'Connection Refused': "Connection-level error — outside WordPress/Agent 3 scope, needs infra investigation.",
    'Request Timeout': "Request timeout — outside WordPress/Agent 3 scope, needs infra investigation.",
    'SSL/TLS Error': "SSL/TLS certificate error — outside WordPress/Agent 3 scope, needs infra investigation.",
    'Connection Error': "Connection-level error — outside WordPress/Agent 3 scope, needs infra investigation.",
    'Slow Response Time': "Server response time — likely infra/caching, outside Agent 3's WordPress-content scope.",
    'Moderate Response Time': "Server response time — likely infra/caching, outside Agent 3's WordPress-content scope.",

    # 3b — access isn't the gap; the gap is judgment or blast-radius
    'Thin Content': lambda ticket: (
        f"{ticket.get('details', 'Page content is below the recommended word count')} — rewriting "
        f"requires human judgment about what this page should actually say; Agent 3 doesn't have the "
        f"domain context to author replacement copy safely."
    ),
    'Duplicate Content Detected': lambda ticket: (
        f"{ticket.get('details', 'Content overlaps with another page')} — needs a human to confirm "
        f"whether this overlap is intentional (e.g. syndicated or boilerplate content) before any "
        f"consolidation or rewrite."
    ),
    'Noindex Tag Present': "Confirmed sitewide cause: WP Admin → Settings → Reading → "
        "'Discourage search engines from indexing this site' is checked. This is a "
        "one-click admin setting, not a per-page fix — toggle it off there if indexing "
        "should resume.",
    'Nofollow Tag Present': lambda ticket: (
        f"{ticket.get('details', 'Page has a nofollow directive')} — nofollow may be a deliberate "
        f"editorial choice (e.g. this page links to third-party or sponsored content); needs human "
        f"confirmation before Agent 3 removes it."
    ),
    'Missing Viewport Meta Tag': "Theme/template-level fix (header.php) — out of scope for per-post Agent 3 fixes.",
    'Missing Language Attribute': "Theme/template-level fix (header.php) — out of scope for per-post Agent 3 fixes.",
    'No Structured Data': "Page has no JSON-LD or Schema.org markup — the correct schema type "
        "(Article, Event, Product, etc.) depends on page content and is a judgment call Agent 3 "
        "doesn't make automatically; flagged as a possible future fix-map entry once a "
        "type-detection rule is defined.",
    'Large Page Size': _defer_page_size,
    'Moderate Page Size': _defer_page_size,
    'Broken Image (No Response)': _defer_broken_image,
}

# Issue names that embed a dynamic status code (e.g. "404 Client Error", "Broken Image (404)") —
# matched by substring since the exact string varies per ticket.
DEFER_PATTERNS = [
    (' Server Error', "Server-side error — outside WordPress/Agent 3 scope, needs infra investigation."),
    (' Client Error', "Page doesn't exist — content needs to be authored by a human, not generated by Agent 3."),
    (' Redirect', "Redirect chain — needs a human to confirm the correct final destination before collapsing."),
    ('Broken Image (', _defer_broken_image),
]


def _defer_reason(ticket):
    """Return the defer reason for an issue type Agent 3 can't fix, or None if it's not a defer case.
    Entries in DEFER_REASONS/DEFER_PATTERNS may be a static string or a callable(ticket) for reasons
    that interpolate ticket details or re-check something live."""
    issue = ticket['issue']
    entry = DEFER_REASONS.get(issue)
    if entry is None:
        for pattern, pattern_entry in DEFER_PATTERNS:
            if pattern in issue:
                entry = pattern_entry
                break
    if entry is None:
        return None
    return entry(ticket) if callable(entry) else entry


def run_fix(ticket):
    """Apply a fix for one ticket: {url, issue, details}.
    Returns a result dict describing what happened — used for logging/demo output.
    """
    url   = ticket['url']
    issue = ticket['issue']

    print(f"[Agent 3] Received ticket: '{issue}' on {url}")

    if issue in TITLE_ISSUES:
        print("[Agent 3] Resolving WordPress content object...")
        try:
            wp_object = resolve_wp_object(url)
        except Exception as e:
            print(f"[Agent 3] Could not resolve WordPress object — request to WordPress failed: {e}")
            return {'status': 'error',
                    'reason': (f"Agent 3 attempted to resolve this URL to a WordPress post or page "
                               f"via the REST API, but the request to WordPress failed ({e}) — needs "
                               f"investigation into WordPress access for this site."),
                    'issue': issue, 'url': url}

        if wp_object is None:
            print(f"[Agent 3] No WordPress post/page found for {url} — skipping.")
            return {'status': 'skipped',
                    'reason': ("Agent 3 attempted to resolve this URL to a WordPress post or page "
                               "via the REST API, but found no match — likely a dynamically-generated, "
                               "filtered, or archive-style URL rather than editable content."),
                    'issue': issue, 'url': url}

        print(f"[Agent 3] Found {wp_object['type']} ID {wp_object['id']}.")
        new_title, old_title = fix_title_core(ticket)
        print(f"[Agent 3] Applying fix: updating title on {wp_object['type']} {wp_object['id']}...")

        try:
            apply_fix(url, wp_object['id'], wp_object['type'], 'title', new_title)
        except Exception:
            print("[Agent 3] Could not apply fix — WordPress write failed.")
            return {'status': 'error',
                    'reason': (f"Agent 3 resolved this to {wp_object['type']} ID {wp_object['id']} and "
                               f"generated a new title ({new_title!r}), but the write to WordPress failed."),
                    'issue': issue, 'url': url}

        missing = _confirm_rendered(url, {'__title__': new_title})
        if missing:
            print(f"[Agent 3] Write succeeded but title did not render: {missing}")
            return {'status': 'error', 'reason': f'write succeeded but did not render: {missing}', 'issue': issue, 'url': url}

        rendered_title = _fetch_page_context(url)['title']
        if len(rendered_title) > 60:
            print(f"[Agent 3] Title rendered but total length ({len(rendered_title)}) still exceeds 60 chars: {rendered_title!r}")
            return {
                'status': 'error',
                'reason': (f'title write succeeded and rendered, but total rendered length is '
                           f'{len(rendered_title)} chars (still over 60) — likely an SEO plugin/theme '
                           f'title-template interaction: {rendered_title!r}'),
                'issue': issue,
                'url': url,
            }

        print(f"[Agent 3] Fix applied and confirmed rendered. New title: {new_title}")
        return {
            'status': 'fixed',
            'issue': issue,
            'url': url,
            'object_id': wp_object['id'],
            'object_type': wp_object['type'],
            'meta': {'title': new_title, 'previous_title': old_title},
            'caveat': (f"AI-generated title, applied and confirmed live on the page. Please verify it "
                       f"reads correctly.<ul><li>Before: {html.escape(old_title)!r}</li>"
                       f"<li>After: {html.escape(new_title)!r}</li></ul>"
                       f"Check {_wp_admin_check_link(wp_object['url'], wp_object['type'], wp_object['id'])}"),
        }

    if issue == 'Images Without Alt Text':
        print(f"[Agent 3] Checking images without alt text on {url}...")
        return fix_images_alt_text(ticket)

    if issue == 'Missing H1 Tag':
        print(f"[Agent 3] Looking for a heading candidate on {url}...")
        return fix_missing_h1(ticket)

    if ' Redirect' in issue:
        print(f"[Agent 3] Tracing redirect chain for {url}...")
        final_url = _redirect_confidence_check(url)
        if not final_url:
            reason = _defer_reason(ticket)
            print(f"[Agent 3] Redirect chain isn't a safe single-hop same-domain collapse — deferring. {reason}")
            return {'status': 'skipped', 'reason': reason, 'issue': issue, 'url': url}

        print(f"[Agent 3] Safe to collapse: {url} -> {final_url}")
        namespaces = probe_site(url)['namespaces']
        plugin_slug = decide_required_plugin(issue, namespaces)
        caveat = None
        if plugin_slug == 'redirection' and 'redirection/v1' not in namespaces:
            print(f"[Agent 3] Redirection plugin not found on {url} — attempting to install '{plugin_slug}'...")
            try:
                ensure_plugin_active(url, plugin_slug)
                print(f"[Agent 3] '{plugin_slug}' installed and active.")
                caveat = f"Installed and activated '{plugin_slug}' on this site to apply this fix — please confirm it's appropriate."
            except Exception as e:
                print(f"[Agent 3] Could not auto-install '{plugin_slug}': {e}")
                return {'status': 'skipped',
                        'reason': (f"Agent 3 confirmed this redirect is safe to collapse to {final_url}, "
                                   f"found the Redirection plugin wasn't active, and attempted to "
                                   f"auto-install it — but that failed ({e})."),
                        'issue': issue, 'url': url}

        try:
            create_redirect_rule(url, urlparse(url).path, final_url)
        except Exception:
            print("[Agent 3] Could not apply fix — create_redirect_rule failed.")
            return {'status': 'error',
                    'reason': (f"Agent 3 confirmed this redirect is safe to collapse to {final_url} and "
                               f"attempted to create the redirect rule via the Redirection plugin, but "
                               f"the write failed."),
                    'issue': issue, 'url': url}

        print(f"[Agent 3] Fix applied. Collapsed redirect to {final_url}.")
        return {'status': 'fixed', 'issue': issue, 'url': url, 'meta': {'collapsed_to': final_url}, 'caveat': caveat}

    fix_fn = FIX_MAP.get(issue)
    if not fix_fn:
        reason = _defer_reason(ticket)
        if reason:
            print(f"[Agent 3] '{issue}' is not fixable by Agent 3 — deferring to human. {reason}")
            return {'status': 'skipped', 'reason': reason, 'issue': issue, 'url': url}
        print(f"[Agent 3] '{issue}' is not in the fix map — skipping.")
        return {'status': 'skipped',
                'reason': (f"Agent 3 checked its fix map and defer-reason list for '{issue}' — this issue "
                           f"type isn't covered by either yet, so no automated action was attempted. "
                           f"Needs human review."),
                'issue': issue, 'url': url}

    print("[Agent 3] Resolving WordPress content object...")
    try:
        wp_object = resolve_wp_object(url)
    except Exception as e:
        print(f"[Agent 3] Could not resolve WordPress object — request to WordPress failed: {e}")
        return {'status': 'error',
                'reason': (f"Agent 3 attempted to resolve this URL to a WordPress post or page via "
                           f"the REST API, but the request to WordPress failed ({e}) — needs investigation "
                           f"into WordPress access for this site."),
                'issue': issue, 'url': url}

    if wp_object is None:
        print(f"[Agent 3] No WordPress post/page found for {url} — skipping.")
        return {'status': 'skipped',
                'reason': ("Agent 3 attempted to resolve this URL to a WordPress post or page via "
                           "the REST API, but found no match — likely a dynamically-generated, "
                           "filtered, or archive-style URL rather than editable content."),
                'issue': issue, 'url': url}

    object_id = wp_object['id']
    object_type = wp_object['type']
    print(f"[Agent 3] Found {object_type} ID {object_id}.")

    namespaces = probe_site(url)['namespaces']
    plugin_slug = decide_required_plugin(issue, namespaces)
    install_caveat = None
    if plugin_slug == 'seo-by-rank-math' and 'rankmath/v1' not in namespaces:
        print(f"[Agent 3] RankMath not found on {url} — attempting to install '{plugin_slug}'...")
        try:
            ensure_plugin_active(url, plugin_slug)
            print(f"[Agent 3] '{plugin_slug}' installed and active.")
            install_caveat = f"Installed and activated '{plugin_slug}' on this site to apply this fix — please confirm it's appropriate."
        except Exception as e:
            print(f"[Agent 3] Could not auto-install '{plugin_slug}': {e}")
            return {'status': 'skipped',
                    'reason': (f"Agent 3 resolved this to {object_type} ID {object_id}, found RankMath "
                               f"wasn't active on this site, and attempted to auto-install it — but "
                               f"that failed ({e})."),
                    'issue': issue, 'url': url}

    # Each FIX_MAP entry now returns (meta_dict, caveat) — caveat is None for deterministic
    # fixes (e.g. canonical self-reference) and an AI-generated-content notice for anything
    # an LLM wrote (description/OG/Twitter), so the ticket always says which parts to check
    # by eye vs. which are mechanical.
    meta_dict, fn_caveat = fix_fn(ticket)
    print(f"[Agent 3] Applying fix: updating {list(meta_dict.keys())} on {object_type} {object_id}...")

    try:
        result = apply_rankmath_meta(url, object_id, meta_dict)
    except Exception:
        print("[Agent 3] Could not apply fix — WordPress write failed.")
        return {
            'status': 'error',
            'reason': (f"Agent 3 resolved this to {object_type} ID {object_id} and generated "
                       f"{list(meta_dict.keys())}, but the write to WordPress (RankMath) failed."),
            'issue': issue,
            'url': url,
            'object_id': object_id,
            'object_type': object_type,
        }

    if 'rank_math_canonical_url' in meta_dict:
        canonical_issue = _confirm_canonical_rendered(url, meta_dict['rank_math_canonical_url'])
        if canonical_issue:
            print(f"[Agent 3] Write succeeded but canonical did not render correctly: {canonical_issue}")
            return {
                'status': 'error',
                'reason': (f"{canonical_issue} Check "
                           f"{_wp_admin_check_link(wp_object['url'], object_type, object_id)}"),
                'issue': issue,
                'url': url,
                'object_id': object_id,
                'object_type': object_type,
            }

    if 'rank_math_description' in meta_dict:
        description_issue = _confirm_meta_description_rendered(url, meta_dict['rank_math_description'])
        if description_issue:
            print(f"[Agent 3] Write succeeded but description did not render correctly: {description_issue}")
            return {
                'status': 'error',
                'reason': (f"{description_issue} Check "
                           f"{_wp_admin_check_link(wp_object['url'], object_type, object_id)}"),
                'issue': issue,
                'url': url,
                'object_id': object_id,
                'object_type': object_type,
            }

    print(f"[Agent 3] Fix applied. Updated: {meta_dict}")
    location_note = f"Check {_wp_admin_check_link(wp_object['url'], object_type, object_id)}"
    full_caveat = ' '.join(part for part in (install_caveat, fn_caveat, location_note) if part)
    return {
        'status': 'fixed',
        'issue': issue,
        'url': url,
        'object_id': object_id,
        'object_type': object_type,
        'meta': meta_dict,
        'result': result,
        'caveat': full_caveat,
    }


def add_ticket_comment(ticket_id, project, comment):
    """Post a comment on an Azure DevOps work item.
    Used to explain to a human why Agent 3 deferred a ticket without changing its state.
    Returns True on success, False on failure (logs the failure).
    """
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')

    token   = base64.b64encode(f':{pat}'.encode()).decode()
    api_url = f'https://dev.azure.com/{org}/{quote(project)}/_apis/wit/workItems/{ticket_id}/comments?api-version=7.1-preview.3'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {token}',
    }
    body = {'text': comment}

    try:
        resp = requests.post(api_url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        print(f"[Ticket] Comment added to ticket {ticket_id}.")
        return True
    except Exception:
        print(f"[Ticket] Could not add comment to ticket {ticket_id} — request to Azure DevOps failed.")
        return False


def set_ticket_tag(ticket_id, tag):
    """Add a tag to an Azure DevOps work item's System.Tags field without disturbing
    any tags already there. System.Tags is a single semicolon-separated string, and a
    PATCH 'add' on it REPLACES the whole field rather than appending — so the current
    value has to be read first and the new tag merged in, or this would silently wipe
    out any tag (e.g. AZURE_QA_TAG) already on the ticket.
    Returns True on success, False on failure (logs the failure) or if the tag was
    already present (no write needed).
    """
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    ticket_id = int(ticket_id)

    token   = base64.b64encode(f':{pat}'.encode()).decode()
    api_url = f'https://dev.azure.com/{org}/_apis/wit/workitems/{ticket_id}?api-version=7.1'
    headers = {'Authorization': f'Basic {token}'}

    try:
        resp = requests.get(f'{api_url}&fields=System.Tags', headers=headers, timeout=10)
        resp.raise_for_status()
        current_tags = resp.json().get('fields', {}).get('System.Tags', '')
    except Exception:
        print(f"[Ticket] Could not read tags on ticket {ticket_id} — request to Azure DevOps failed.")
        return False

    existing = [t.strip() for t in current_tags.split(';') if t.strip()]
    if tag in existing:
        return False

    merged_tags = '; '.join(existing + [tag])
    patch_headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': f'Basic {token}',
    }
    body = [{'op': 'add', 'path': '/fields/System.Tags', 'value': merged_tags}]

    try:
        resp = requests.patch(api_url, headers=patch_headers, json=body, timeout=10)
        resp.raise_for_status()
        print(f"[Ticket] Tag '{tag}' added to ticket {ticket_id}.")
        return True
    except Exception:
        print(f"[Ticket] Could not add tag to ticket {ticket_id} — request to Azure DevOps failed.")
        return False


def set_ticket_state(ticket_id, state):
    """Update an Azure DevOps work item's System.State field.
    Used to flag a ticket as ready for manual QA after Agent 3 applies a fix.
    Returns True on success, False on failure (logs the failure).
    """
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    ticket_id = int(ticket_id)

    token   = base64.b64encode(f':{pat}'.encode()).decode()
    api_url = f'https://dev.azure.com/{org}/_apis/wit/workitems/{ticket_id}?api-version=7.1'
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': f'Basic {token}',
    }
    body = [{'op': 'add', 'path': '/fields/System.State', 'value': state}]

    try:
        resp = requests.patch(api_url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        print(f"[Ticket] Ticket {ticket_id} moved to '{state}'.")
        return True
    except Exception:
        print(f"[Ticket] Could not update ticket {ticket_id} state — request to Azure DevOps failed.")
        return False
