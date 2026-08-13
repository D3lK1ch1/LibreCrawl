import threading
import time
import csv
import json
import xml.etree.ElementTree as ET
import uuid
import webbrowser
import argparse
import secrets
import string
import os
import requests
import re
from io import StringIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_compress import Compress
from functools import wraps
from src.crawler import WebCrawler
from src.settings_manager import SettingsManager
from src.agents.ticket_review_agent import run_agentic as run_review_agent
from src.agents.provider import get_provider, set_provider_override, ANTHROPIC_MODEL, OPENAI_MODEL, ANTHROPIC_EXPLAIN_MODEL, OPENAI_EXPLAIN_MODEL, record_usage, get_usage_summary, get_usage_log
from src.agents.prompts import (
    build_explain_issue_prompt, EXPLAIN_ISSUE_SYSTEM_PROMPT,
    build_agent_chat_prompt, AGENT_CHAT_SYSTEM_PROMPT,
)
from src.auth_db import init_db, create_user, authenticate_user, get_user_by_id, log_guest_crawl, get_guest_crawls_last_24h, verify_user, set_user_tier, create_verification_token, verify_token, get_user_by_email, create_magic_link, verify_magic_link
from src.email_service import send_verification_email, send_welcome_email, send_magic_link_email

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# OpenAI client
openai_client = None
openai_api_key = (os.getenv('OPENAI_API_KEY') or '').strip()
if openai_api_key:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_api_key)
    except ImportError:
        print("Warning: openai package not installed.")
    except Exception:
        print("Warning: OpenAI client could not be initialized.")

# Anthropic client
anthropic_client = None
anthropic_api_key = (os.getenv('ANTHROPIC_API_KEY') or '').strip()
if anthropic_api_key:
    try:
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=anthropic_api_key)
    except ImportError:
        print("Warning: anthropic package not installed.")
    except Exception:
        print("Warning: Anthropic client could not be initialized.")

# Parse command line arguments
parser = argparse.ArgumentParser(description='LibreCrawl - SEO Spider Tool')
parser.add_argument('--local', '-l', action='store_true',
                    help='Run in local mode (all users get admin tier, no rate limits)')
parser.add_argument('--disable-register', '-dr', action='store_true',
                    help='Disable new user registrations')
parser.add_argument('--disable-guest', '-dg', action='store_true',
                    help='Disable guest login')
parser.add_argument('--demo', '-dm', action='store_true',
                    help='Demo mode: 1.5GB memory limit per user, crawls auto-stop at limit')
parser.add_argument('--dangerously-skip-auth', '-dsa', action='store_true',
                    help='DANGEROUS: Allow anyone to log in as any username with no password. '
                         'The username is only used to separate per-user sessions. '
                         'Do NOT use on a public network or in production.')
args = parser.parse_args()

LOCAL_MODE = args.local
DISABLE_REGISTER = args.disable_register
DISABLE_GUEST = args.disable_guest or os.getenv('DISABLE_GUEST', '').lower() in ('true', '1', 'yes')
DEMO_MODE = args.demo or os.getenv('DEMO_MODE', '').lower() in ('true', '1', 'yes')
SKIP_AUTH = args.dangerously_skip_auth or os.getenv('DANGEROUSLY_SKIP_AUTH', '').lower() in ('true', '1', 'yes')
ALLOWED_EMAIL_DOMAIN = os.getenv('ALLOWED_EMAIL_DOMAIN', '')
MAIN_APP_URL = os.getenv('MAIN_APP_URL', 'http://localhost:5000').rstrip('/')

AGENT2_ENABLED = os.getenv('AGENT2_ENABLED', 'true').lower() != 'false'
AGENT3_ENABLED = os.getenv('AGENT3_ENABLED', 'true').lower() != 'false'
AGENT4_ENABLED = os.getenv('AGENT4_ENABLED', 'true').lower() != 'false'
AZURE_QA_TAG   = os.getenv('AZURE_QA_TAG', 'qa-agent')
AZURE_FIX_TAG  = os.getenv('AZURE_FIX_TAG', 'fix-agent')

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.secret_key = 'librecrawl-secret-key-change-in-production'  # TODO: Use environment variable in production

# Enable compression for all responses
Compress(app)

# Initialize database on startup
init_db()

def _safe_error(e, where):
    """Log the real exception server-side and return a generic, client-safe message.
    Returning str(e) (or a raw upstream response body) directly in an API response can
    leak internal details — file paths, hostnames, stack traces, Azure DevOps error
    bodies — to whoever calls the endpoint. See CodeQL 'information exposure through
    an exception' (py/stack-trace-exposure)."""
    print(f"[Error] {where}: {e}")
    return 'An internal error occurred. Check server logs for details.'

_PROJECT_NAME_RE = re.compile(r'^[A-Za-z0-9 ._-]{1,64}$')

_AZURE_PROJECTS_CACHE = {}  # org -> (fetched_at, {names})
_AZURE_PROJECTS_CACHE_TTL = 300  # seconds; avoids one Azure call per ticket in a bulk batch

def _get_valid_azure_project_names(org, pat):
    """Live set of real project names in this Azure org, cached briefly. Returns None
    (not an empty set) if the Azure API call itself fails, so callers can fall back to
    format-only validation instead of wrongly rejecting every project during an Azure
    outage or PAT issue unrelated to the request being validated."""
    cached = _AZURE_PROJECTS_CACHE.get(org)
    now = time.time()
    if cached and now - cached[0] < _AZURE_PROJECTS_CACHE_TTL:
        return cached[1]
    import base64
    try:
        token = base64.b64encode(f':{pat}'.encode()).decode()
        resp = requests.get(
            f'https://dev.azure.com/{org}/_apis/projects?api-version=7.1&$top=100',
            headers={'Authorization': f'Basic {token}'}, timeout=10
        )
        resp.raise_for_status()
        names = {p['name'] for p in resp.json().get('value', [])}
    except Exception as e:
        print(f"[Warn] _get_valid_azure_project_names: {e}")
        return None
    _AZURE_PROJECTS_CACHE[org] = (now, names)
    return names

def _validate_project_name(project):
    """Azure DevOps project names get interpolated directly into REST URL paths.
    Layer 1 (always, no network): reject anything outside the character set Azure
    actually allows for a project name. Layer 2 (best-effort): confirm it's a project
    this org's Azure account actually has right now, via a briefly-cached live lookup —
    real semantic validation instead of format-shape alone. Falls back to layer 1 only
    if the live lookup is unavailable, rather than breaking ticket creation on an
    unrelated Azure API hiccup. Raises ValueError if invalid."""
    if not project or not _PROJECT_NAME_RE.match(project):
        raise ValueError(f'invalid Azure DevOps project name: {project!r}')
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    if org and pat:
        valid_names = _get_valid_azure_project_names(org, pat)
        if valid_names is not None and project not in valid_names:
            raise ValueError(f'invalid Azure DevOps project name: {project!r}')

def generate_random_password(length=16):
    """Generate a random password with letters, digits, and symbols"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def auto_login_local_mode():
    """Auto-login for local mode - creates or logs into 'local' admin account"""
    import sqlite3
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'users.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if 'local' user exists
        cursor.execute('SELECT id, username, tier FROM users WHERE username = ?', ('local',))
        user = cursor.fetchone()

        if user:
            # User exists, just log them in
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['tier'] = 'admin'
            session.permanent = True
            print(f"Auto-logged in as existing 'local' user (ID: {user['id']})")
        else:
            # Create new local user with random password
            random_password = generate_random_password()
            from src.auth_db import hash_password
            password_hash = hash_password(random_password)

            cursor.execute('''
                INSERT INTO users (username, email, password_hash, verified, tier)
                VALUES (?, ?, ?, 1, 'admin')
            ''', ('local', 'local@localhost', password_hash))
            conn.commit()

            user_id = cursor.lastrowid

            # Log in the new user
            session['user_id'] = user_id
            session['username'] = 'local'
            session['tier'] = 'admin'
            session.permanent = True

            print(f"Created and auto-logged in as new 'local' admin user (ID: {user_id})")
            print(f"Generated password: {random_password}")

        conn.close()
        return True
    except Exception as e:
        print(f"Error in auto_login_local_mode: {e}")
        return False

def skip_auth_login(username):
    """Skip-auth login: create user record if missing, log them in.

    Each username gets its own user_id, which drives per-user crawler
    instance and settings isolation. No password is checked. Always
    grants admin tier (matches local-mode behavior).

    Returns (success, message).
    """
    import sqlite3
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'users.db'))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT id, username FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()

        if user:
            user_id = user['id']
        else:
            from src.auth_db import hash_password
            random_password = generate_random_password()
            password_hash = hash_password(random_password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, verified, tier)
                VALUES (?, ?, ?, 1, 'admin')
            ''', (username, f'{username}@skipauth.local', password_hash))
            conn.commit()
            user_id = cursor.lastrowid

        conn.close()

        session['user_id'] = user_id
        session['username'] = username
        session['tier'] = 'admin'
        session.permanent = True

        return True, 'Logged in (authentication skipped)'
    except sqlite3.IntegrityError as e:
        # Most likely the generated email collides with an existing account
        # whose email happens to match. Fall back to a clearer message.
        return False, 'Username conflict: try a different username'
    except Exception as e:
        print(f"Error in skip_auth_login: {e}")
        return False, _safe_error(e, 'skip_auth_login')

if LOCAL_MODE:
    print("=" * 60)
    print("LOCAL MODE ENABLED")
    print("All users will have admin tier access")
    print("No rate limits or tier restrictions")
    print("Auto-login enabled with 'local' admin account")
    print("=" * 60)

if DISABLE_REGISTER:
    print("=" * 60)
    print("REGISTRATION DISABLED")
    print("New user registrations are not allowed")
    print("=" * 60)

if DISABLE_GUEST:
    print("=" * 60)
    print("GUEST MODE DISABLED")
    print("Guest login is not allowed")
    print("=" * 60)

if DEMO_MODE:
    print("=" * 60)
    print("DEMO MODE ENABLED")
    print("Memory limit: 1.5GB per user")
    print("Crawls will auto-stop when limit is reached")
    print("=" * 60)

if SKIP_AUTH:
    print("=" * 60)
    print("⚠️  DANGEROUSLY SKIP AUTH ENABLED")
    print("Anyone can log in as any username with no password!")
    print("Username is used only to separate per-user sessions.")
    print("DO NOT use on a public network or production server!")
    print("=" * 60)

def get_client_ip():
    """Get the real client IP address, checking Cloudflare headers first"""
    # Check Cloudflare header first
    if 'CF-Connecting-IP' in request.headers:
        return request.headers['CF-Connecting-IP']
    # Check other common proxy headers
    if 'X-Forwarded-For' in request.headers:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    if 'X-Real-IP' in request.headers:
        return request.headers['X-Real-IP']
    # Fall back to direct connection IP
    return request.remote_addr

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In local mode, auto-login if not already logged in
        if LOCAL_MODE and 'user_id' not in session:
            auto_login_local_mode()
        elif 'user_id' not in session:
            # Not in local mode and not logged in
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Authentication required'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Multi-tenant crawler instances
crawler_instances = {}  # session_id -> {'crawler': WebCrawler, 'settings': SettingsManager, 'last_accessed': datetime}
instances_lock = threading.Lock()

def get_or_create_crawler():
    """Get or create a crawler instance for the current session"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']
    user_id = session.get('user_id')
    tier = session.get('tier', 'guest')

    with instances_lock:
        if session_id not in crawler_instances:
            # Before creating a blank instance, adopt any crawl that is still running.
            # This handles the case where the app restarted and the session cookie was
            # invalidated, leaving a live crawl thread orphaned under a different session_id.
            for existing_id, instance in crawler_instances.items():
                if instance['crawler'].is_running:
                    session['session_id'] = existing_id
                    instance['last_accessed'] = datetime.now()
                    print(f"Session {session_id} adopted running crawl from session {existing_id}")
                    return instance['crawler']

            print(f"Creating new crawler instance for session: {session_id}, user: {user_id}, tier: {tier}")
            crawler_instances[session_id] = {
                'crawler': WebCrawler(),
                'settings': SettingsManager(session_id=session_id, user_id=user_id, tier=tier),
                'last_accessed': datetime.now()
            }
        else:
            crawler_instances[session_id]['last_accessed'] = datetime.now()

        return crawler_instances[session_id]['crawler']

def get_session_settings():
    """Get the settings manager for the current session"""
    # Get or create session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']
    user_id = session.get('user_id')  # Get user_id from session
    tier = session.get('tier', 'guest')  # Get tier from session

    with instances_lock:
        # Create instance if it doesn't exist
        if session_id not in crawler_instances:
            print(f"Creating new settings instance for session: {session_id}, user: {user_id}, tier: {tier}")
            crawler_instances[session_id] = {
                'crawler': WebCrawler(),
                'settings': SettingsManager(session_id=session_id, user_id=user_id, tier=tier),
                'last_accessed': datetime.now()
            }
        else:
            # Update last accessed time
            crawler_instances[session_id]['last_accessed'] = datetime.now()

        return crawler_instances[session_id]['settings']

def cleanup_old_instances():
    """Remove crawler instances that haven't been accessed in 1 hour"""
    timeout = timedelta(hours=1)
    now = datetime.now()

    with instances_lock:
        sessions_to_remove = []
        for session_id, instance_data in crawler_instances.items():
            if now - instance_data['last_accessed'] > timeout:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            print(f"Cleaning up crawler instance for session: {session_id}")
            # Stop any running crawls
            try:
                crawler_instances[session_id]['crawler'].stop_crawl()
            except:
                pass
            del crawler_instances[session_id]

        if sessions_to_remove:
            print(f"Cleaned up {len(sessions_to_remove)} inactive crawler instances")

def start_cleanup_thread():
    """Start background thread to cleanup old instances"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Check every 5 minutes
            try:
                cleanup_old_instances()
            except Exception as e:
                print(f"Error in cleanup thread: {e}")

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("Started crawler instance cleanup thread")

def generate_csv_export(urls, fields):
    """Generate CSV export content"""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    for url_data in urls:
        row = {}
        for field in fields:
            value = url_data.get(field, '')

            # Handle complex data types for CSV
            if field == 'analytics' and isinstance(value, dict):
                analytics_list = []
                if value.get('gtag') or value.get('ga4_id'): analytics_list.append('GA4')
                if value.get('google_analytics'): analytics_list.append('GA')
                if value.get('gtm_id'): analytics_list.append('GTM')
                if value.get('facebook_pixel'): analytics_list.append('FB')
                if value.get('hotjar'): analytics_list.append('HJ')
                if value.get('mixpanel'): analytics_list.append('MP')
                row[field] = ', '.join(analytics_list)
            elif field == 'og_tags' and isinstance(value, dict):
                row[field] = f"{len(value)} tags" if value else ''
            elif field == 'twitter_tags' and isinstance(value, dict):
                row[field] = f"{len(value)} tags" if value else ''
            elif field == 'json_ld' and isinstance(value, list):
                row[field] = f"{len(value)} scripts" if value else ''
            elif field == 'images' and isinstance(value, list):
                row[field] = f"{len(value)} images" if value else ''
            elif field == 'internal_links' and isinstance(value, (int, float)):
                row[field] = f"{int(value)} internal links" if value else '0 internal links'
            elif field == 'external_links' and isinstance(value, (int, float)):
                row[field] = f"{int(value)} external links" if value else '0 external links'
            elif field == 'h2' and isinstance(value, list):
                row[field] = ', '.join(value[:3]) + ('...' if len(value) > 3 else '')
            elif field == 'h3' and isinstance(value, list):
                row[field] = ', '.join(value[:3]) + ('...' if len(value) > 3 else '')
            elif isinstance(value, (dict, list)):
                row[field] = str(value)
            else:
                row[field] = value

        writer.writerow(row)

    return output.getvalue()

def generate_json_export(urls, fields):
    """Generate JSON export content"""
    filtered_urls = []
    for url_data in urls:
        filtered_data = {}
        for field in fields:
            value = url_data.get(field, '')
            # Keep complex data structures intact in JSON
            filtered_data[field] = value
        filtered_urls.append(filtered_data)

    return json.dumps({
        'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_urls': len(filtered_urls),
        'fields': fields,
        'data': filtered_urls
    }, indent=2, default=str)

def generate_xml_export(urls, fields):
    """Generate XML export content"""
    root = ET.Element('librecrawl_export')
    root.set('export_date', time.strftime('%Y-%m-%d %H:%M:%S'))
    root.set('total_urls', str(len(urls)))

    urls_element = ET.SubElement(root, 'urls')

    for url_data in urls:
        url_element = ET.SubElement(urls_element, 'url')
        for field in fields:
            field_element = ET.SubElement(url_element, field)
            field_element.text = str(url_data.get(field, ''))

    return ET.tostring(root, encoding='unicode')

def generate_links_csv_export(links):
    """Generate CSV export for links data"""
    output = StringIO()
    fieldnames = ['source_url', 'target_url', 'anchor_text', 'is_internal', 'target_domain', 'target_status', 'placement']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for link in links:
        row = {
            'source_url': link.get('source_url', ''),
            'target_url': link.get('target_url', ''),
            'anchor_text': link.get('anchor_text', ''),
            'is_internal': 'Yes' if link.get('is_internal') else 'No',
            'target_domain': link.get('target_domain', ''),
            'target_status': link.get('target_status', 'Not crawled'),
            'placement': link.get('placement', 'body')
        }
        writer.writerow(row)

    return output.getvalue()

def generate_links_json_export(links):
    """Generate JSON export for links data"""
    return json.dumps(links, indent=2)

def filter_issues_by_exclusion_patterns(issues, exclusion_patterns):
    """Filter issues based on exclusion patterns (applies current settings to loaded crawls)"""
    from fnmatch import fnmatch
    from urllib.parse import urlparse

    if not exclusion_patterns:
        return issues

    filtered_issues = []

    for issue in issues:
        url = issue.get('url', '')
        parsed = urlparse(url)
        path = parsed.path

        # Check if URL matches any exclusion pattern
        should_exclude = False
        for pattern in exclusion_patterns:
            if not pattern.strip() or pattern.strip().startswith('#'):
                continue

            if '*' in pattern:
                if fnmatch(path, pattern):
                    should_exclude = True
                    break
            elif path == pattern or path.startswith(pattern.rstrip('*')):
                should_exclude = True
                break

        if not should_exclude:
            filtered_issues.append(issue)

    return filtered_issues

def _extract_asset_url(details): 
    match = re.search(r'(?:Image returned \d+|Image does not respond):\s*(.+)', details or '')
    return match.group(1).strip() if match else '' 

def generate_issues_csv_export(issues, crawl_id=None, user_id=None):
    """Generate CSV export for issues data"""

    from src.crawl_db import get_crawl_by_id, get_issue_first_detected_bulk

    crawl = get_crawl_by_id(crawl_id) if crawl_id else {}
    crawl_date = crawl.get('completed_at', '') if crawl else ''
    started_at = crawl.get('started_at', '') if crawl else ''

    pairs = [{'url': issue.get('url', ''), 'issue': issue.get('issue', '')} for issue in issues]
    date_lookup = get_issue_first_detected_bulk(pairs, user_id) if user_id else {}

    output = StringIO()
    fieldnames = ['url', 'type', 'category', 'issue', 'details', 'affected_asset_url', 'crawl_date', 'first_detected', 'is_new']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    seen = set()
    deduped = []

    for issue in issues:
        key = issue.get('url', '') + '|' + issue.get('issue', '')
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    for issue in deduped:
        key = issue.get('url', '') + '|' + issue.get('issue', '')
        first_detected = date_lookup.get(key, '')
        row = {
            'url': issue.get('url', ''),
            'type': issue.get('type', ''),
            'category': issue.get('category', ''),
            'issue': issue.get('issue', ''),
            'details': issue.get('details', ''),
            'affected_asset_url': _extract_asset_url(issue.get('details', '')),
            'crawl_date': crawl_date,
            'first_detected': first_detected,
            'is_new': 'Yes' if first_detected and started_at and first_detected >= started_at else 'No'
        }
        writer.writerow(row)

    return output.getvalue()

def generate_issues_json_export(issues):
    """Generate JSON export for issues data"""
    # Group issues by URL for better organization
    issues_by_url = {}
    for issue in issues:
        url = issue.get('url', '')
        if url not in issues_by_url:
            issues_by_url[url] = []
        issues_by_url[url].append({
            'type': issue.get('type', ''),
            'category': issue.get('category', ''),
            'issue': issue.get('issue', ''),
            'details': issue.get('details', '')
        })

    return json.dumps({
        'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_issues': len(issues),
        'total_urls_with_issues': len(issues_by_url),
        'issues_by_url': issues_by_url,
        'all_issues': issues
    }, indent=2)

@app.route('/api/probe_http_errors', methods=['POST'])
@login_required
def probe_http_errors():
    """API endpoint to probe HTTP errors for a list of URLs"""
    data = request.get_json()
    issues = data.get('issues', [])

    from src.agents.url_safety import safe_head, UnsafeURLError

    resolved_urls = []
    BROKEN_IMAGE_PATTERNS = ['Broken Image']
    headers = {'User-Agent': 'LibreCrawlBot/1.0 (+https://librecrawl.com)'}

    for issue in issues:
        issue_name = issue.get('issue', '')
        is_image = any(pattern in issue_name for pattern in BROKEN_IMAGE_PATTERNS)
        probe_url = _extract_asset_url(issue.get('details', '')) if is_image else issue.get('url', '')

        if not probe_url:
            continue
        try:
            response = safe_head(probe_url, allow_redirects=True, timeout=10, headers=headers)
            status_code = response.status_code
            if status_code < 400:
                resolved_urls.append({'url': issue.get('url', ''), 'issue': issue_name})
        except UnsafeURLError:
            continue
        except Exception as e:
            pass

    return jsonify({'success': True, 'resolved_urls': resolved_urls})

@app.route('/login')
def login_page():
    # In local mode, auto-login and redirect to index
    if LOCAL_MODE:
        auto_login_local_mode()
        return redirect(url_for('index'))
    # Redirect to app if already logged in
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html', registration_disabled=DISABLE_REGISTER, guest_disabled=DISABLE_GUEST, skip_auth=SKIP_AUTH, allowed_domain=ALLOWED_EMAIL_DOMAIN)

@app.route('/register')
def register_page():
    # Redirect to app if already logged in
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('register.html', registration_disabled=DISABLE_REGISTER)

@app.route('/verify')
def verify_email():
    """Email verification endpoint"""
    token = request.args.get('token')

    if not token:
        return render_template('verification_result.html',
                             success=False,
                             message='Invalid verification link',
                             app_source='main')

    # Verify the token
    success, message, app_source, user_email = verify_token(token)

    # Send welcome email if successful
    if success and user_email:
        try:
            user = get_user_by_email(user_email)
            if user:
                send_welcome_email(user_email, user['username'], app_source or 'main')
        except Exception as e:
            print(f"Error sending welcome email: {e}")

    # Determine redirect URL based on app_source
    redirect_url = None
    if success:
        if app_source == 'workshop':
            redirect_url = os.getenv('WORKSHOP_APP_URL', 'https://workshop.librecrawl.com')
        else:
            redirect_url = url_for('login_page')

    return render_template('verification_result.html',
                         success=success,
                         message=message,
                         app_source=app_source or 'main',
                         redirect_url=redirect_url)

@app.route('/api/request-magic-link', methods=['POST'])
def request_magic_link():
    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()

    if not email or '@' not in email:
        return jsonify({'success': False, 'message': 'A valid email address is required.'})

    if ALLOWED_EMAIL_DOMAIN and not email.endswith(f'@{ALLOWED_EMAIL_DOMAIN}'):
        return jsonify({'success': False, 'message': f'Only @{ALLOWED_EMAIL_DOMAIN} addresses are allowed.'})

    token = create_magic_link(email)
    if not token:
        return jsonify({'success': False, 'message': 'Failed to generate login link. Please try again.'})

    magic_url = f"{MAIN_APP_URL}/auth/magic?token={token}"
    send_magic_link_email(email, magic_url)
    return jsonify({'success': True, 'message': 'Check your email for a login link.'})


@app.route('/auth/magic', methods=['GET'])
def magic_link_auth():
    token = request.args.get('token', '').strip()
    if not token:
        return redirect(url_for('login_page', error='invalid'))

    success, user_id, message = verify_magic_link(token)

    if not success:
        return redirect(url_for('login_page', error='invalid'))

    user = get_user_by_id(user_id)
    session['user_id'] = user_id
    session['username'] = user['username']
    session['tier'] = 'admin' if LOCAL_MODE else user['tier']
    session.permanent = True
    return redirect(url_for('index'))


@app.route('/api/register', methods=['POST'])
def register():
    return jsonify({'success': False, 'message': 'Registration is not available. Use magic link login.'}), 410

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    # Dangerously skip auth: accept any username with no password.
    if SKIP_AUTH:
        if not username:
            return jsonify({'success': False, 'message': 'Username required'})
        if len(username) > 50:
            return jsonify({'success': False, 'message': 'Username must be 50 characters or less'})
        success, message = skip_auth_login(username)
        return jsonify({'success': success, 'message': message})

    return jsonify({'success': False, 'message': 'Password login is not available. Use OTP login.'}), 410

@app.route('/api/guest-login', methods=['POST'])
def guest_login():
    """Login as a guest user (no account required, limited to 3 crawls/24h)"""
    if DISABLE_GUEST:
        return jsonify({'success': False, 'message': 'Guest login is disabled'})

    # Create a guest session with no user_id but with tier='guest'
    # In local mode, guests also get admin tier
    session['user_id'] = None
    session['username'] = 'Guest'
    session['tier'] = 'admin' if LOCAL_MODE else 'guest'
    session.permanent = False  # Don't persist guest sessions

    return jsonify({'success': True, 'message': 'Logged in as guest'})

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/user/info')
@login_required
def user_info():
    """Get current user info including tier"""
    from src.auth_db import get_crawls_last_24h
    user_id = session.get('user_id')
    tier = session.get('tier', 'guest')
    username = session.get('username')

    # Get crawl count
    crawls_today = 0
    if tier == 'guest':
        # For guests, count from IP address
        client_ip = get_client_ip()
        crawls_today = get_guest_crawls_last_24h(client_ip)
    else:
        # For registered users, count from database
        crawls_today = get_crawls_last_24h(user_id)

    return jsonify({
        'success': True,
        'user': {
            'id': user_id,
            'username': username,
            'tier': tier,
            'crawls_today': crawls_today,
            'crawls_remaining': max(0, 3 - crawls_today) if tier == 'guest' else -1
        }
    })

@app.route('/')
def index():
    # In local mode, auto-login if not already logged in
    if LOCAL_MODE and 'user_id' not in session:
        auto_login_local_mode()
    elif 'user_id' not in session:
        # Not in local mode and not logged in, redirect to login
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Crawl history dashboard"""
    return render_template('dashboard.html')

@app.route('/debug/memory')
@login_required
def debug_memory_page():
    """Debug page with nice UI for memory monitoring"""
    return render_template('debug_memory.html')

@app.route('/api/start_crawl', methods=['POST'])
@login_required
def start_crawl():
    from src.auth_db import get_crawls_last_24h, log_crawl_start

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'success': False, 'error': 'URL is required'})

    user_id = session.get('user_id')
    tier = session.get('tier', 'guest')

    # Check guest limits (IP-based) - skip in local mode
    if tier == 'guest' and not LOCAL_MODE:
        client_ip = get_client_ip()
        crawls_from_ip = get_guest_crawls_last_24h(client_ip)

        if crawls_from_ip >= 3:
            return jsonify({
                'success': False,
                'error': 'Guest limit reached: 3 crawls per 24 hours from your IP address. Please register for unlimited crawls.'
            })

        # Log this guest crawl
        log_guest_crawl(client_ip)

    # Get or create crawler for this session
    crawler = get_or_create_crawler()
    session_id = session.get('session_id')
    settings_manager = get_session_settings()

    # Apply current settings to crawler before starting
    try:
        crawler_config = settings_manager.get_crawler_config()
        crawler.update_config(crawler_config)
    except Exception as e:
        print(f"Warning: Could not apply settings: {e}")

    # Enforce demo mode limits
    if DEMO_MODE:
        crawler.config['demo_mode'] = True
        crawler.config['demo_memory_limit_bytes'] = int(1.5 * 1024 * 1024 * 1024)  # 1.5GB

    # Pass user_id and session_id for database persistence
    success, message = crawler.start_crawl(url, user_id=user_id, session_id=session_id)

    # Store crawl_id in session
    if success and crawler.crawl_id:
        session['current_crawl_id'] = crawler.crawl_id
        # Also log to old crawl_history for compatibility
        log_crawl_start(user_id, url)

    return jsonify({'success': success, 'message': message, 'crawl_id': crawler.crawl_id})

@app.route('/api/stop_crawl', methods=['POST'])
@login_required
def stop_crawl():
    crawler = get_or_create_crawler()
    success, message = crawler.stop_crawl()
    return jsonify({'success': success, 'message': message})

@app.route('/api/crawl_status')
@login_required
def crawl_status():
    crawler = get_or_create_crawler()
    settings_manager = get_session_settings()

    # Check for incremental update parameters
    url_since = request.args.get('url_since', type=int)
    link_since = request.args.get('link_since', type=int)
    issue_since = request.args.get('issue_since', type=int)

    # Get full status data
    status_data = crawler.get_status()

    # Ensure baseUrl is in stats (needed for UI to work correctly)
    if crawler.base_url and 'stats' in status_data:
        status_data['stats']['baseUrl'] = crawler.base_url

    # Check if we need to force a full refresh (after loading from DB)
    force_full = session.pop('force_full_refresh', False)

    # If incremental parameters provided AND not forcing full refresh, slice the arrays
    if not force_full:
        if url_since is not None:
            status_data['urls'] = status_data.get('urls', [])[url_since:]
        if link_since is not None:
            status_data['links'] = status_data.get('links', [])[link_since:]
        if issue_since is not None:
            status_data['issues'] = status_data.get('issues', [])[issue_since:]

    # Apply current issue exclusion patterns to displayed issues
    issues = status_data.get('issues', [])
    if issues:
        current_settings = settings_manager.get_settings()
        exclusion_patterns_text = current_settings.get('issueExclusionPatterns', '')
        exclusion_patterns = [p.strip() for p in exclusion_patterns_text.split('\n') if p.strip()]
        filtered_issues = filter_issues_by_exclusion_patterns(issues, exclusion_patterns)
        status_data['issues'] = filtered_issues

    return jsonify(status_data)

@app.route('/api/visualization_data')
@login_required
def visualization_data():
    """Get graph data for site structure visualization"""
    try:
        crawler = get_or_create_crawler()
        status_data = crawler.get_status()

        # Get URLs from the status data
        crawled_pages = status_data.get('urls', [])
        all_links = status_data.get('links', [])

        # Build nodes and edges for the graph
        nodes = []
        edges = []
        url_to_id = {}

        # Create nodes from crawled pages (limit to prevent lag)
        max_nodes = 500  # Optimization: limit nodes for performance
        pages_to_visualize = crawled_pages[:max_nodes]

        for idx, page in enumerate(pages_to_visualize):
            url = page.get('url', '')
            status_code = page.get('status_code', 0)

            # Assign color based on status code
            if 200 <= status_code < 300:
                color = '#10b981'  # Green for 2xx
            elif 300 <= status_code < 400:
                color = '#3b82f6'  # Blue for 3xx
            elif 400 <= status_code < 500:
                color = '#f59e0b'  # Orange for 4xx
            elif 500 <= status_code < 600:
                color = '#ef4444'  # Red for 5xx
            else:
                color = '#6b7280'  # Gray for other

            # Create node
            node = {
                'data': {
                    'id': f'node-{idx}',
                    'label': url.split('/')[-1] or url.split('//')[-1],  # Use last path segment or domain
                    'url': url,
                    'status_code': status_code,
                    'title': page.get('title', ''),
                    'color': color,
                    'size': 30 if idx == 0 else 20,  # Make root node larger
                    'depth': page.get('depth', 0)
                }
            }
            nodes.append(node)
            url_to_id[url] = f'node-{idx}'

        # Create edges from links data
        # Links are stored as: {'source_url': url, 'target_url': url, 'is_internal': bool, ...}
        edges_set = set()  # Use set to avoid duplicate edges
        for link in all_links:
            if link.get('is_internal'):  # Only use internal links
                source_url = link.get('source_url', '')
                target_url = link.get('target_url', '')

                source_id = url_to_id.get(source_url)
                target_id = url_to_id.get(target_url)

                if source_id and target_id and source_id != target_id:
                    edge_key = f'{source_id}-{target_id}'
                    if edge_key not in edges_set:
                        edges_set.add(edge_key)
                        edge = {
                            'data': {
                                'id': f'edge-{edge_key}',
                                'source': source_id,
                                'target': target_id
                            }
                        }
                        edges.append(edge)

        return jsonify({
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'total_pages': len(crawled_pages),
            'visualized_pages': len(nodes),
            'truncated': len(crawled_pages) > max_nodes
        })

    except Exception as e:
        print(f"Error generating visualization data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': _safe_error(e, 'visualization_data'),
            'nodes': [],
            'edges': []
        })

@app.route('/api/debug/memory')
@login_required
def debug_memory():
    """Debug endpoint showing memory stats for all active crawler instances"""
    with instances_lock:
        memory_stats = {
            'total_instances': len(crawler_instances),
            'instances': []
        }

        for session_id, instance_data in crawler_instances.items():
            crawler = instance_data['crawler']
            stats = crawler.memory_monitor.get_stats()

            memory_stats['instances'].append({
                'session_id': session_id[:8] + '...',  # Truncate for privacy
                'last_accessed': instance_data['last_accessed'].isoformat(),
                'urls_crawled': len(crawler.crawl_results),
                'memory': stats,
                'data_sizes': crawler.user_memory.get_stats()
            })

        return jsonify(memory_stats)

@app.route('/api/debug/memory/profile')
@login_required
def debug_memory_profile():
    """Detailed memory profiling - what's actually using the RAM"""
    from src.core.memory_profiler import MemoryProfiler

    with instances_lock:
        profiles = []

        for session_id, instance_data in crawler_instances.items():
            crawler = instance_data['crawler']

            # Get object breakdown
            breakdown = MemoryProfiler.get_object_memory_breakdown()

            profiles.append({
                'session_id': session_id[:8] + '...',
                'urls_crawled': len(crawler.crawl_results),
                'object_breakdown': breakdown,
                'data_sizes': crawler.user_memory.get_stats()
            })

        return jsonify({
            'total_instances': len(crawler_instances),
            'profiles': profiles
        })

@app.route('/api/filter_issues', methods=['POST'])
@login_required
def filter_issues():
    try:
        data = request.get_json()
        issues = data.get('issues', [])
        settings_manager = get_session_settings()

        # Get current exclusion patterns
        current_settings = settings_manager.get_settings()
        exclusion_patterns_text = current_settings.get('issueExclusionPatterns', '')
        exclusion_patterns = [p.strip() for p in exclusion_patterns_text.split('\n') if p.strip()]

        # Filter issues
        filtered_issues = filter_issues_by_exclusion_patterns(issues, exclusion_patterns)

        return jsonify({'success': True, 'issues': filtered_issues})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'filter_issues')})

@app.route('/api/get_settings')
@login_required
def get_settings():
    try:
        settings_manager = get_session_settings()
        settings = settings_manager.get_settings()
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'get_settings')})

@app.route('/api/save_settings', methods=['POST'])
@login_required
def save_settings():
    try:
        data = request.get_json()
        settings_manager = get_session_settings()
        success, message = settings_manager.save_settings(data)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'save_settings')})

@app.route('/api/reset_settings', methods=['POST'])
@login_required
def reset_settings():
    try:
        settings_manager = get_session_settings()
        success, message = settings_manager.reset_settings()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'reset_settings')})

@app.route('/api/update_crawler_settings', methods=['POST'])
@login_required
def update_crawler_settings():
    try:
        crawler = get_or_create_crawler()
        settings_manager = get_session_settings()
        # Get current settings and update crawler configuration
        crawler_config = settings_manager.get_crawler_config()
        crawler.update_config(crawler_config)
        return jsonify({'success': True, 'message': 'Crawler settings updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'update_crawler_settings')})

@app.route('/api/pause_crawl', methods=['POST'])
@login_required
def pause_crawl():
    try:
        crawler = get_or_create_crawler()
        success, message = crawler.pause_crawl()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'pause_crawl')})

@app.route('/api/resume_crawl', methods=['POST'])
@login_required
def resume_crawl():
    try:
        crawler = get_or_create_crawler()
        success, message = crawler.resume_crawl()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'resume_crawl')})

@app.route('/api/crawls/list')
@login_required
def list_crawls():
    """Get all crawls for current user"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import get_user_crawls, get_crawl_count

        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status_filter = request.args.get('status')

        crawls = get_user_crawls(user_id, limit=limit, offset=offset, status_filter=status_filter)
        total_count = get_crawl_count(user_id)

        return jsonify({
            'success': True,
            'crawls': crawls,
            'total': total_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'list_crawls')})

@app.route('/api/crawls/<int:crawl_id>')
@login_required
def get_crawl(crawl_id):
    """Get complete crawl data by ID"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import get_crawl_by_id, load_crawled_urls, load_crawl_links, load_crawl_issues

        # Get crawl metadata
        crawl = get_crawl_by_id(crawl_id)
        if not crawl:
            return jsonify({'success': False, 'error': 'Crawl not found'}), 404

        # Check ownership (guests have user_id = None)
        if user_id and crawl.get('user_id') != user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        # Load all data
        urls = load_crawled_urls(crawl_id)
        links = load_crawl_links(crawl_id)
        issues = load_crawl_issues(crawl_id)

        return jsonify({
            'success': True,
            'crawl': crawl,
            'urls': urls,
            'links': links,
            'issues': issues
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': _safe_error(e, 'get_crawl')}), 500

@app.route('/api/crawls/<int:crawl_id>/load', methods=['POST'])
@login_required
def load_crawl_into_session(crawl_id):
    """Load a historical crawl into the current session"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import get_crawl_by_id, load_crawled_urls, load_crawl_links, load_crawl_issues

        # Get crawl metadata
        crawl = get_crawl_by_id(crawl_id)
        if not crawl:
            return jsonify({'success': False, 'error': 'Crawl not found'}), 404

        # Check ownership
        if user_id and crawl.get('user_id') != user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        # Get current crawler instance
        crawler = get_or_create_crawler()

        # Stop any running crawl
        if crawler.is_running:
            crawler.stop_crawl()

        # Load all data from database
        urls = load_crawled_urls(crawl_id)
        links = load_crawl_links(crawl_id)
        issues = load_crawl_issues(crawl_id)

        # Inject into current crawler instance
        with crawler.results_lock:
            crawler.crawl_results = urls
            crawler.stats['crawled'] = len(urls)
            crawler.stats['discovered'] = len(urls)
            crawler.base_url = crawl['base_url']
            crawler.base_domain = crawl['base_domain']

        # Load links into link manager
        if crawler.link_manager:
            crawler.link_manager.all_links = links
            # Rebuild links_set
            crawler.link_manager.links_set.clear()
            for link in links:
                link_key = f"{link['source_url']}|{link['target_url']}"
                crawler.link_manager.links_set.add(link_key)

        # Load issues into issue detector
        if crawler.issue_detector:
            crawler.issue_detector.detected_issues = issues

        # Rebuild per-user memory tracker for loaded data
        crawler.user_memory.reset()
        crawler._demo_limit_reached = False
        for url_data in urls:
            crawler.user_memory.track_url(url_data)
        if links:
            crawler.user_memory.track_links(links)
        if issues:
            crawler.user_memory.track_issues(issues)

        # Set Flask session flag for force full refresh
        session['force_full_refresh'] = True
        # Without this, create_bulk_tickets' crawl_issues_exist() check would validate
        # against whatever crawl_id was last set (crawl-start/resume) instead of the
        # crawl actually being reviewed here — either wrongly rejecting real approvals
        # or cross-referencing the wrong crawl's issues.
        session['current_crawl_id'] = crawl_id

        return jsonify({
            'success': True,
            'message': f'Loaded {len(urls)} URLs, {len(links)} links, {len(issues)} issues',
            'urls_count': len(urls),
            'links_count': len(links),
            'issues_count': len(issues),
            'should_refresh_ui': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': _safe_error(e, 'load_crawl_into_session')}), 500

@app.route('/api/crawls/<int:crawl_id>/resume', methods=['POST'])
@login_required
def resume_crawl_endpoint(crawl_id):
    """Resume an interrupted crawl"""
    try:
        user_id = session.get('user_id')
        session_id = session.get('session_id')

        # Get crawler for this session
        crawler = get_or_create_crawler()

        # Enforce demo mode limits on resumed crawls
        if DEMO_MODE:
            crawler.config['demo_mode'] = True
            crawler.config['demo_memory_limit_bytes'] = int(1.5 * 1024 * 1024 * 1024)

        # Resume from database
        success, message = crawler.resume_from_database(crawl_id, user_id=user_id, session_id=session_id)

        if success:
            session['current_crawl_id'] = crawl_id

        return jsonify({'success': success, 'message': message})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': _safe_error(e, 'resume_crawl_endpoint')})

@app.route('/api/crawls/<int:crawl_id>/delete', methods=['DELETE'])
@login_required
def delete_crawl_endpoint(crawl_id):
    """Delete a crawl and all associated data"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import delete_crawl, get_crawl_by_id

        # Verify ownership
        crawl = get_crawl_by_id(crawl_id)
        if not crawl:
            return jsonify({'success': False, 'error': 'Crawl not found'}), 404

        if user_id and crawl.get('user_id') != user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        success = delete_crawl(crawl_id)
        return jsonify({'success': success, 'message': 'Crawl deleted successfully' if success else 'Failed to delete crawl'})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'delete_crawl_endpoint')})

@app.route('/api/crawls/<int:crawl_id>/archive', methods=['POST'])
@login_required
def archive_crawl(crawl_id):
    """Archive crawl (mark as archived but keep data)"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import set_crawl_status, get_crawl_by_id

        # Verify ownership
        crawl = get_crawl_by_id(crawl_id)
        if not crawl:
            return jsonify({'success': False, 'error': 'Crawl not found'}), 404

        if user_id and crawl.get('user_id') != user_id:
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

        success = set_crawl_status(crawl_id, 'archived')
        return jsonify({'success': success, 'message': 'Crawl archived successfully' if success else 'Failed to archive crawl'})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'archive_crawl')})

@app.route('/api/crawls/stats')
@login_required
def crawl_stats():
    """Get statistics about user's crawls"""
    try:
        user_id = session.get('user_id')
        from src.crawl_db import get_crawl_count, get_database_size_mb
        import sqlite3

        # Get counts by status
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'users.db'))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT status, COUNT(*) as count
            FROM crawls
            WHERE user_id = ?
            GROUP BY status
        ''', (user_id,))

        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return jsonify({
            'success': True,
            'total_crawls': get_crawl_count(user_id),
            'by_status': status_counts,
            'database_size_mb': get_database_size_mb()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'crawl_stats')})

@app.route('/api/export_data', methods=['POST'])
@login_required
def export_data():
    try:
        data = request.get_json()
        export_format = data.get('format', 'csv')
        export_fields = data.get('fields', ['url', 'status_code', 'title'])
        local_data = data.get('localData', {})

        # Use local data if provided (from loaded crawl), otherwise get from crawler
        if local_data and local_data.get('urls'):
            urls = local_data.get('urls', [])
            links = local_data.get('links', [])
            issues = local_data.get('issues', [])
        else:
            # Get current crawl results
            crawler = get_or_create_crawler()
            crawl_data = crawler.get_status()
            urls = crawl_data.get('urls', [])
            links = crawl_data.get('links', [])
            issues = crawl_data.get('issues', [])

        if not urls:
            return jsonify({'success': False, 'error': 'No data to export'})

        # Update link statuses from crawled URLs (fixes missing status codes in exports)
        if links and urls:
            status_lookup = {url_data['url']: url_data.get('status_code') for url_data in urls}
            for link in links:
                target_url = link.get('target_url')
                if target_url in status_lookup:
                    link['target_status'] = status_lookup[target_url]

        # Apply current issue exclusion patterns (works for loaded crawls too)
        if issues:
            settings_manager = get_session_settings()
            current_settings = settings_manager.get_settings()
            exclusion_patterns_text = current_settings.get('issueExclusionPatterns', '')
            exclusion_patterns = [p.strip() for p in exclusion_patterns_text.split('\n') if p.strip()]
            issues = filter_issues_by_exclusion_patterns(issues, exclusion_patterns)

        # Collect files to export based on special field selections
        files_to_export = []

        # Check for special export fields and prepare them as separate files
        has_issues_export = 'issues_detected' in export_fields
        has_links_export = 'links_detailed' in export_fields

        # Remove special fields from regular export fields
        regular_fields = [f for f in export_fields if f not in ['issues_detected', 'links_detailed']]

        # Generate issues export if requested
        if has_issues_export:
            if export_format == 'csv':
                issues_content = generate_issues_csv_export(issues, crawl_id = session.get('current_crawl_id'), user_id = session.get('user_id'))
                issues_mimetype = 'text/csv'
                issues_filename = f'librecrawl_issues_{int(time.time())}.csv'
            elif export_format == 'json':
                issues_content = generate_issues_json_export(issues)
                issues_mimetype = 'application/json'
                issues_filename = f'librecrawl_issues_{int(time.time())}.json'
            else:
                issues_content = generate_issues_csv_export(issues, crawl_id = session.get('current_crawl_id'), user_id = session.get('user_id'))
                issues_mimetype = 'text/csv'
                issues_filename = f'librecrawl_issues_{int(time.time())}.csv'

            files_to_export.append({
                'content': issues_content,
                'mimetype': issues_mimetype,
                'filename': issues_filename
            })

        # Generate links export if requested
        if has_links_export:
            if export_format == 'csv':
                links_content = generate_links_csv_export(links)
                links_mimetype = 'text/csv'
                links_filename = f'librecrawl_links_{int(time.time())}.csv'
            elif export_format == 'json':
                links_content = generate_links_json_export(links)
                links_mimetype = 'application/json'
                links_filename = f'librecrawl_links_{int(time.time())}.json'
            else:
                links_content = generate_links_csv_export(links)
                links_mimetype = 'text/csv'
                links_filename = f'librecrawl_links_{int(time.time())}.csv'

            files_to_export.append({
                'content': links_content,
                'mimetype': links_mimetype,
                'filename': links_filename
            })

        # Generate regular export if there are regular fields
        if regular_fields:
            if export_format == 'csv':
                regular_content = generate_csv_export(urls, regular_fields)
                regular_mimetype = 'text/csv'
                regular_filename = f'librecrawl_export_{int(time.time())}.csv'
            elif export_format == 'json':
                regular_content = generate_json_export(urls, regular_fields)
                regular_mimetype = 'application/json'
                regular_filename = f'librecrawl_export_{int(time.time())}.json'
            elif export_format == 'xml':
                regular_content = generate_xml_export(urls, regular_fields)
                regular_mimetype = 'application/xml'
                regular_filename = f'librecrawl_export_{int(time.time())}.xml'
            else:
                return jsonify({'success': False, 'error': 'Unsupported export format'})

            files_to_export.append({
                'content': regular_content,
                'mimetype': regular_mimetype,
                'filename': regular_filename
            })

        # Handle special case where only special fields are selected but no data
        if not files_to_export:
            if has_issues_export and not issues:
                return jsonify({'success': False, 'error': 'No issues data to export'})
            elif has_links_export and not links:
                return jsonify({'success': False, 'error': 'No links data to export'})
            else:
                return jsonify({'success': False, 'error': 'No data to export'})

        # Return multiple files if we have more than one, otherwise single file
        if len(files_to_export) > 1:
            return jsonify({
                'success': True,
                'multiple_files': True,
                'files': files_to_export
            })
        else:
            # Single file
            file_data = files_to_export[0]
            return jsonify({
                'success': True,
                'content': file_data['content'],
                'mimetype': file_data['mimetype'],
                'filename': file_data['filename']
            })

    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'export_data')})

def recover_crashed_crawls():
    """Check for and recover any crashed crawls on startup"""
    try:
        from src.crawl_db import get_crashed_crawls, set_crawl_status

        crashed = get_crashed_crawls()

        if crashed:
            print("\n" + "=" * 60)
            print("CRASH RECOVERY")
            print("=" * 60)
            for crawl in crashed:
                set_crawl_status(crawl['id'], 'failed')
                print(f"Found crashed crawl: {crawl['base_url']} (ID: {crawl['id']})")
                print(f"  → Marked as failed. User can resume from dashboard.")
            print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error during crash recovery: {e}")

def graceful_shutdown(signum, frame):
    """Save all active crawls before shutdown"""
    print("\n" + "=" * 60)
    print("GRACEFUL SHUTDOWN")
    print("=" * 60)
    print("Saving all active crawls...")

    try:
        with instances_lock:
            for session_id, instance_data in list(crawler_instances.items()):
                crawler = instance_data['crawler']
                if crawler.is_running and crawler.crawl_id and crawler.db_save_enabled:
                    print(f"  → Saving crawl {crawler.crawl_id}...")
                    try:
                        crawler._save_batch_to_db(force=True)
                        crawler._save_queue_checkpoint()
                        from src.crawl_db import set_crawl_status
                        set_crawl_status(crawler.crawl_id, 'paused')
                    except Exception as e:
                        print(f"    Error saving crawl {crawler.crawl_id}: {e}")

        print("All crawls saved successfully")
        print("=" * 60)
    except Exception as e:
        print(f"Error during shutdown: {e}")

    print("Goodbye!")
    import sys
    sys.exit(0)

CATEGORY_ROLE_MAP = {
    'Technical':        'Web Developer',
    'Performance':      'Web Developer',
    'SEO':              'Webmaster',
    'Content':          'Copywriter / Content Editor',
    'Accessibility':    'Web Developer / Designer',
    'Mobile':           'Web Developer / Designer',
    'Social':           'Copywriter / Content Editor',
    'Structured Data':  'Webmaster',
    'Indexability':     'Webmaster',
}

@app.route('/api/explain_issue', methods=['POST'])
@login_required
def explain_issue():
    """Generate AI-powered explanation and fix for a crawl issue using OpenAI"""
    try:
        # Check if OpenAI and Anthropic client is available
        provider = get_provider()
        if provider is None:
            return jsonify({
                'success': False,
                'error': 'No AI provider configured'
            }), 400

        data = request.get_json()
        url = data.get('url', '')
        issue = data.get('issue', '')
        category = data.get('category', '')
        details = data.get('details', '')
        page_context = data.get('page_context', {})
        # Defaults to agent2 (this route's usual caller); qa_agent.py passes
        # 'agent4' so its calls don't get misattributed to Agent 2's token log.
        caller_agent = data.get('agent', 'agent2')

        # Build context string from page data
        context_parts = []
        if page_context.get('title'):
            context_parts.append(f"Page title: {page_context['title']}")
        if page_context.get('word_count'):
            context_parts.append(f"Word count: {page_context['word_count']}")
        if page_context.get('meta_description'):
            context_parts.append(f"Meta description: {page_context['meta_description'][:100]}")
        if page_context.get('h1'):
            context_parts.append(f"H1: {page_context['h1']}")

        context_str = '\n'.join(context_parts) if context_parts else 'No additional context available'

        # Prompt built in src/agents/prompts.py — see build_explain_issue_prompt()
        prompt = build_explain_issue_prompt(url, issue, category, details, context_str)

        # Call Anthropic API
        if provider == 'anthropic':
            response = anthropic_client.messages.create(
                model=ANTHROPIC_EXPLAIN_MODEL,
                system=EXPLAIN_ISSUE_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=500,
            )
        else:
        # Call OpenAI API
            response = openai_client.chat.completions.create(
            model=OPENAI_EXPLAIN_MODEL,
            messages=[
                {'role': 'system', 'content': EXPLAIN_ISSUE_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            response_format={'type': 'json_object'}
        )

        # Parse response
        if provider == 'openai':
            ai_response = json.loads(response.choices[0].message.content)
        else:
            text = response.content[0].text
            ai_response = json.loads(text[text.find('{'):text.rfind('}')+1])

        # Log token usage
        explain_model = ANTHROPIC_EXPLAIN_MODEL if provider == 'anthropic' else OPENAI_EXPLAIN_MODEL
        if provider == 'anthropic':
            input_tokens, output_tokens = response.usage.input_tokens, response.usage.output_tokens
        else:
            input_tokens, output_tokens = response.usage.prompt_tokens, response.usage.completion_tokens
        tokens = input_tokens + output_tokens
        record_usage(caller_agent, explain_model, input_tokens, output_tokens, label=issue)
        print(f"AI Explain - Tokens used: {tokens}")

        how_to_fix = ai_response.get('how_to_fix', '')
        if isinstance(how_to_fix, list):
            how_to_fix = '• ' + '\n• '.join(str(item).strip() for item in how_to_fix if item)

        return jsonify({
            'success': True,
            'explanation': ai_response.get('explanation', ''),
            'how_to_fix': how_to_fix,
            'priority': ai_response.get('priority', 'medium'),
            'role': ai_response.get('role', ''),
            'tokens_used': tokens,
            'model': explain_model
        })

    except Exception as e:
        print(f"AI Explain Error: {e}")
        return jsonify({
            'success': False,
            'error': _safe_error(e, 'explain_issue')
        }), 500

def _create_single_ticket(url, issue, category, issue_type, ai_explanation, ai_how_to_fix, ai_priority, ai_role, project, parent_id, assignee=None, user_id=None):
    """Shared ticket creation logic used by both the single-issue route and bulk agent route."""
    import base64
    from urllib.parse import urlparse, quote

    org      = os.getenv('AZURE_DEVOPS_ORG')
    pat      = os.getenv('AZURE_DEVOPS_PAT')
    sm_email = os.getenv('AZURE_DEVOPS_SM_EMAIL')

    missing = [k for k, v in {'AZURE_DEVOPS_ORG': org, 'AZURE_DEVOPS_PAT': pat, 'AZURE_DEVOPS_SM_EMAIL': sm_email}.items() if not v]
    if missing:
        return False, {'error': f'Missing .env variables: {", ".join(missing)}'}
    try:
        _validate_project_name(project)
    except ValueError as e:
        return False, {'error': str(e)}

    if issue_type == 'error':
        az_priority, sup_label, moscow = (1, 'Critical', 'Must') if ai_priority == 'high' else (2, 'High', 'Should')
    elif issue_type == 'warning':
        az_priority, sup_label, moscow = 3, 'Warning', 'Could'
    else:
        az_priority, sup_label, moscow = 4, 'Informational', "Won't"

    valid_roles = {'Webmaster', 'Copywriter / Content Editor', 'Web Developer', 'Designer'}
    role      = ai_role if ai_role in valid_roles else CATEGORY_ROLE_MAP.get(category, 'Web Developer')
    fix_items = [p.strip() for p in ai_how_to_fix.split('•') if p.strip()]
    fix_html  = '<ul>' + ''.join(f'<li>{item}</li>' for item in fix_items) + '</ul>' \
                if fix_items else f'<p>{ai_how_to_fix}</p>'

    parsed    = urlparse(url)
    short_url = (parsed.path.rstrip('/') or parsed.netloc)[-60:]

    description_html = (
        f'<h3>🧩 Summary</h3>'
        f'<p>{ai_explanation} This issue was detected by LibreCrawl on '
        f'<a href="{url}">{url}</a>. '
        f'Category: {category}. Severity: {sup_label} — Priority {az_priority} ({moscow}).</p>'
        f'<h3>👥 Responsibility</h3>'
        f'<ul><li>{role} — required</li>'
        f'<li>Scrum Master to review and delegate based on role above</li></ul>'
        f'<h3>⚙️ Implementation Direction</h3>'
        f'{fix_html}'
    )

    ac_html = (
        f'<ul>'
        f'<li>Issue "{issue}" should no longer be detected by LibreCrawl on {url}</li>'
        f'<li>Page should pass {category} validation in the next scheduled crawl</li>'
        f'<li>Verified and closed within sprint by Scrum Master</li>'
        f'</ul>'
    )

    title     = f'[{category}] {issue} — {short_url}'
    token     = base64.b64encode(f':{pat}'.encode()).decode()
    work_type = quote('Product Backlog Item')
    api_url   = f'https://dev.azure.com/{org}/{quote(project)}/_apis/wit/workitems/${work_type}?api-version=7.1'

    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': f'Basic {token}',
    }
    body = [
        {'op': 'add', 'path': '/fields/System.Title',                             'value': title},
        {'op': 'add', 'path': '/fields/System.Description',                       'value': description_html},
        {'op': 'add', 'path': '/fields/Microsoft.VSTS.Common.Priority',           'value': az_priority},
        {'op': 'add', 'path': '/fields/System.AssignedTo',                        'value': assignee or sm_email},
        {'op': 'add', 'path': '/fields/Microsoft.VSTS.Common.AcceptanceCriteria', 'value': ac_html},
        {'op': 'add', 'path': '/fields/System.AreaPath',                          'value': f'{project}\\{os.environ.get("AZURE_AREA_SUFFIX")}'},
        {'op': 'add', 'path': '/relations/-', 'value': {
            'rel': 'System.LinkTypes.Hierarchy-Reverse',
            'url': f'https://dev.azure.com/{org}/_apis/wit/workitems/{parent_id}',
            'attributes': {'comment': 'Created by LibreCrawl Page Diagnostics'}
        }},
    ]

    try:
        resp = requests.post(api_url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        ticket_id  = resp.json()['id']
        ticket_url = f'https://dev.azure.com/{org}/{quote(project)}/_workitems/edit/{ticket_id}'
        from src.crawl_db import save_devops_ticket
        save_devops_ticket(url, issue, category, ticket_id, ticket_url, user_id=user_id)
        return True, {'ticket_id': ticket_id, 'ticket_url': ticket_url, 'title': title}
    except requests.exceptions.HTTPError:
        return False, {'error': _safe_error(f'Azure DevOps {resp.status_code}: {resp.text}', '_create_single_ticket')}
    except Exception as e:
        return False, {'error': _safe_error(e, '_create_single_ticket')}


@app.route('/api/create_devops_ticket', methods=['POST'])
@login_required
def create_devops_ticket():
    """Create an Azure DevOps Product Backlog Item from a Page Diagnostics issue"""
    data        = request.get_json()
    url         = data.get('url', '')
    issue       = data.get('issue', '')
    category    = data.get('category', '')
    issue_type  = data.get('issue_type', 'warning')
    ai_exp      = data.get('ai_explanation', '')
    ai_fix      = data.get('ai_how_to_fix', '')
    ai_priority = data.get('ai_priority', 'medium')
    ai_role     = data.get('ai_role', '')
    assignee    = data.get('assignee_override') or None

    project   = data.get('project_override') or os.getenv('AZURE_DEVOPS_PROJECT', '')
    parent_id = data.get('parent_id_override') or os.getenv('AZURE_DEVOPS_PARENT_ID', '')

    if not project:
        return jsonify({'success': False, 'error': 'No Azure project selected. Use the Project dropdown next to the Clear button.'}), 400
    if not parent_id:
        return jsonify({'success': False, 'error': 'No Feature selected. Use the Feature dropdown next to the Clear button.'}), 400

    success, result = _create_single_ticket(
        url, issue, category, issue_type, ai_exp, ai_fix, ai_priority, ai_role,
        project, parent_id, assignee=assignee, user_id=session.get('user_id')
    )
    if success:
        return jsonify({'success': True, **result})
    return jsonify({'success': False, 'error': result['error']}), 500

def _filter_removed_tickets(tickets):
    """Dedup safety net: a locally-stored devops_tickets row can point at a ticket_id whose
    Azure work item has since moved to 'Removed' state — treating that as still 'exists'
    would block a legitimate new ticket forever. Batch-fetches System.State for every
    ticket_id in `tickets` via Azure's org-level work item batch GET (ticket IDs are unique
    per-org; devops_tickets has no project column to scope a project-level call to).
    Removed-state entries are dropped from the returned dict and their stale local rows
    deleted. Fails open: any error talking to Azure returns `tickets` unfiltered — dedup is
    a safety net, not a gate, matching the frontend's existing `.catch(() => {})` philosophy.
    """
    if not tickets:
        return tickets
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    if not org or not pat:
        return tickets
    try:
        import base64
        ticket_ids = list({str(t['ticket_id']) for t in tickets.values()})[:200]
        token = base64.b64encode(f':{pat}'.encode()).decode()
        headers = {'Authorization': f'Basic {token}', 'Content-Type': 'application/json'}
        batch_url = (
            f'https://dev.azure.com/{org}/_apis/wit/workitems'
            f'?ids={",".join(ticket_ids)}&fields=System.Id,System.State&api-version=7.1'
        )
        resp = requests.get(batch_url, headers=headers, timeout=10)
        resp.raise_for_status()
        removed_ids = {
            str(item['id']) for item in resp.json().get('value', [])
            if item.get('fields', {}).get('System.State') == 'Removed'
        }
        if not removed_ids:
            return tickets

        from src.crawl_db import delete_devops_tickets
        delete_devops_tickets([int(tid) for tid in removed_ids])
        return {ck: t for ck, t in tickets.items() if str(t['ticket_id']) not in removed_ids}
    except Exception as e:
        print(f"[Dedup] Could not check Azure ticket states — skipping Removed-state filter: {e}")
        return tickets

@app.route('/api/devops_tickets/check', methods=['POST'])
@login_required
def check_devops_tickets():
    try:
        from src.crawl_db import get_tickets_for_issues
        data = request.get_json()
        pairs = data.get('pairs', [])
        tickets = get_tickets_for_issues(pairs)
        tickets = _filter_removed_tickets(tickets)
        return jsonify({'success': True, 'tickets': tickets})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'check_devops_tickets')})

@app.route('/api/devops/projects', methods=['GET'])
@login_required
def devops_projects():
    import base64
    from urllib.parse import quote
    org = os.getenv('AZURE_DEVOPS_ORG')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    if not org or not pat:
        return jsonify({'success': False, 'error': 'AZURE_DEVOPS_ORG or AZURE_DEVOPS_PAT not configured'}), 400
    token   = base64.b64encode(f':{pat}'.encode()).decode()
    api_url = f'https://dev.azure.com/{org}/_apis/projects?api-version=7.1&$top=100'
    try:
        resp = requests.get(api_url, headers={'Authorization': f'Basic {token}'}, timeout=10)
        resp.raise_for_status()
        projects = [{'id': p['id'], 'name': p['name']} for p in resp.json().get('value', [])]
        return jsonify({'success': True, 'projects': projects})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'devops_projects')}), 500


@app.route('/api/devops/features', methods=['GET'])
@login_required
def devops_features():
    import base64
    from urllib.parse import quote
    org     = os.getenv('AZURE_DEVOPS_ORG')
    pat     = os.getenv('AZURE_DEVOPS_PAT')
    project = request.args.get('project', '')
    if not org or not pat:
        return jsonify({'success': False, 'error': 'AZURE_DEVOPS_ORG or AZURE_DEVOPS_PAT not configured'}), 400
    if not project:
        return jsonify({'success': False, 'error': 'project query param required'}), 400
    try:
        _validate_project_name(project)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    token    = base64.b64encode(f':{pat}'.encode()).decode()
    headers  = {'Authorization': f'Basic {token}', 'Content-Type': 'application/json'}
    wiql_url = f'https://dev.azure.com/{org}/{quote(project)}/_apis/wit/wiql?api-version=7.1&$top=200'
    wiql_body = {'query': (
        "SELECT [System.Id] FROM WorkItems "
        "WHERE [System.WorkItemType] IN ('Epic', 'Feature') "
        "AND [System.TeamProject] = @project "
        "AND [System.State] <> 'Removed' "
        "ORDER BY [System.Title]"
    )}
    try:
        wiql_resp = requests.post(wiql_url, headers=headers, json=wiql_body, timeout=10)
        wiql_resp.raise_for_status()
        ids = [str(item['id']) for item in wiql_resp.json().get('workItems', [])][:200]
        if not ids:
            return jsonify({'success': True, 'features': []})
        batch_url  = (
            f'https://dev.azure.com/{org}/{quote(project)}/_apis/wit/workitems'
            f'?ids={",".join(ids)}&fields=System.Id,System.Title,System.WorkItemType&api-version=7.1'
        )
        batch_resp = requests.get(batch_url, headers=headers, timeout=10)
        batch_resp.raise_for_status()
        features = [
            {
                'id':   item['id'],
                'name': item['fields']['System.Title'],
                'type': item['fields'].get('System.WorkItemType', '')
            }
            for item in batch_resp.json().get('value', [])
        ]
        return jsonify({'success': True, 'features': features})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'devops_features')}), 500

_agent_state = {}
_qa_bulk_state = {'running': False, 'results': None}

@app.route("/api/agent/start_workflow", methods=['POST'])
@login_required
def start_workflow():
    if not AGENT2_ENABLED:
        return jsonify({'success': False, 'error': 'Agent 2 (review) is disabled — set AGENT2_ENABLED=true to enable.'}), 503
    global _agent_state
    data = request.get_json()
    _agent_state["url"]      = data.get("url")
    _agent_state["project"]  = data.get("project")
    _agent_state["feature"]  = data.get("feature")
    # create_bulk_tickets' forged-request guard needs crawl_id, but its own call comes
    # from mcp_server.py's http_session (a separate cookie jar, self-calling localhost) —
    # session.get('current_crawl_id') would be empty for that caller. Captured here
    # instead, same as project/feature above, since this route runs in the browser's own
    # session where current_crawl_id is actually set.
    _agent_state["crawl_id"] = session.get('current_crawl_id')
    _agent_state["status"]   = "Workflow started"
    _agent_state["issues"]  = data.get("issues", [])
    _agent_state["results"] = None

    issues = _agent_state["issues"]
    thread = threading.Thread(target=run_review_agent, args=(issues,), daemon=True)
    thread.start()

    return jsonify({'success': True})

@app.route("/api/agent/workflow_trigger", methods=['GET'])
@login_required
def workflow_trigger():
    global _agent_state
    if _agent_state.get("status") != "Workflow started":
        return jsonify({'success': False, 'error': 'No active workflow'}), 400
    
    _agent_state["status"] = "Crawling"

    return jsonify({'ready': True, 'url': _agent_state.get("url"), 'project': _agent_state.get("project"), 'feature': _agent_state.get("feature"), 'issues': _agent_state.get("issues", [])})

@app.route("/api/agent/review", methods=['POST'])
@login_required
def agent_review():
    global _agent_state
    data = request.get_json()
    _agent_state["review"] = data.get("issues", [])
    _agent_state["status"] = "Review ready"
    return jsonify({'success': True})

@app.route("/api/agent/review", methods=['GET'])
@login_required
def get_review():
    global _agent_state
    if _agent_state.get("status") != "Review ready":
        return jsonify({'success': False, 'error': 'Review not ready'}), 400
    return jsonify({'success': True, 'issues': _agent_state.get("review", [])})

@app.route("/api/agent/approval", methods=['POST'])
@login_required
def agent_approval():
    global _agent_state
    data = request.get_json()
    _agent_state["approval"] = data.get("approval", [])
    _agent_state["status"] = "Approval received"
    return jsonify({'success': True})

@app.route("/api/agent/approval", methods=['GET'])
@login_required
def get_approval():
    global _agent_state
    if _agent_state.get("status") != "Approval received":
        return jsonify({'success': False, 'error': 'Approval not received'}), 400
    return jsonify({'success': True, 'approval': _agent_state.get("approval", [])})

@app.route("/api/agent/create_bulk_tickets", methods=['POST'])
@login_required
def create_bulk_tickets():
    global _agent_state
    data      = request.get_json()
    approved  = data.get("approved", [])
    project   = _agent_state.get("project", "")
    parent_id = _agent_state.get("feature", "")

    crawl_id = _agent_state.get('crawl_id')
    if not crawl_id:
        return jsonify({'success': False, 'error': 'No active crawl in this session — load or resume a crawl first.'}), 400

    import base64 as _b64
    from src.crawl_db import get_tickets_for_issues, crawl_issues_exist

    created = []
    errors  = []

    # Generate cache_keys matching the JS formula used by the plugin:
    # 'ai_' + btoa(unescape(encodeURIComponent(url+'|'+issue))).replace(/[=+/]/g, '')
    # Python equivalent: base64(utf-8 bytes), strip =, +, /
    def _cache_key(u, iss):
        raw = (u + '|' + iss).encode('utf-8')
        return 'ai_' + _b64.b64encode(raw).decode('ascii').replace('=', '').replace('+', '').replace('/', '')

    pairs    = [{'url': i.get('url',''), 'issue': i.get('issue',''), 'cache_key': _cache_key(i.get('url',''), i.get('issue',''))} for i in approved]
    existing = get_tickets_for_issues(pairs)
    existing = _filter_removed_tickets(existing)
    # `approved` is client-supplied JSON — only issues LibreCrawl itself detected during
    # this session's crawl are eligible to become tickets (and reach Agent 3's fetch).
    # Without this, a forged url in the request body would flow straight into run_fix().
    valid_keys = crawl_issues_exist(crawl_id, pairs)

    for i, issue in enumerate(approved):
        url        = issue.get("url", "")
        issue_name = issue.get("issue", "")
        ck         = pairs[i]['cache_key']

        if f"{url}|{issue_name}" not in valid_keys:
            errors.append({'url': url, 'issue': issue_name,
                            'error': 'This issue was not found in the current crawl — rejected.'})
            continue

        if ck in existing:
            t = existing[ck]
            created.append({'ticket_id': t.get('ticket_id'), 'ticket_url': t.get('ticket_url'), 'title': issue_name, 'skipped': True})
            continue

        success, result = _create_single_ticket(
            url,
            issue_name,
            issue.get("category", ""),
            issue.get("type", "warning"),
            issue.get("explanation", ""),
            issue.get("how_to_fix", ""),
            issue.get("priority", "medium"),
            issue.get("role", ""),
            project,
            parent_id,
            assignee=issue.get("assignee") or None,
            user_id=session.get('user_id')
        )
        if success:
            if not AGENT3_ENABLED:
                result['agent3_status'] = 'disabled'
                result['agent3_reason'] = 'Agent 3 is disabled — set AGENT3_ENABLED=true to enable auto-fix.'
            else:
                try:
                    from src.agents.fix_agent import run_fix, set_ticket_state, set_ticket_tag, add_ticket_comment
                    fix_result = run_fix({'url': url, 'issue': issue_name, 'details': issue.get('details', '')})
                    result['agent3_status'] = fix_result.get('status')
                    result['agent3_reason'] = fix_result.get('reason', '')
                    result['agent3_object_id'] = fix_result.get('object_id')
                    result['agent3_object_type'] = fix_result.get('object_type')

                    if fix_result['status'] == 'fixed':
                        qa_state = os.getenv('AZURE_QA_STATE', 'QA')
                        result['agent3_qa_state'] = qa_state
                        result['agent3_qa_updated'] = set_ticket_state(result['ticket_id'], qa_state)
                        result['agent3_tag_added'] = set_ticket_tag(result['ticket_id'], AZURE_FIX_TAG)
                        # Agent 4's QA Bulk Run finds tickets by AZURE_QA_TAG, not by state —
                        # without this, a ticket Agent 3 fixes and moves to 'QA' state never
                        # surfaces in that checklist for Agent 4 to pick up automatically.
                        result['agent3_qa_tag_added'] = set_ticket_tag(result['ticket_id'], AZURE_QA_TAG)
                        if fix_result.get('caveat'):
                            result['agent3_comment_added'] = add_ticket_comment(result['ticket_id'], project, fix_result['caveat'])
                    elif fix_result['status'] in ('skipped', 'error') and fix_result.get('reason'):
                        result['agent3_comment_added'] = add_ticket_comment(result['ticket_id'], project, fix_result['reason'])
                except Exception as e:
                    result['agent3_status'] = 'error'
                    result['agent3_reason'] = _safe_error(e, 'create_bulk_tickets (Agent 3)')

            created.append(result)
        else:
            errors.append({**result, 'url': url})

    _agent_state["results"] = {"created": created, "errors": errors}
    _agent_state["status"]  = "Completed"
    return jsonify({'success': True, 'created': created, 'errors': errors})

@app.route("/api/agent/results", methods=['GET'])
@login_required
def get_results():
    global _agent_state
    if _agent_state.get("status") != "Completed":
        return jsonify({'ready': False})
    return jsonify({'ready': True, 'results': _agent_state.get("results", {})})

@app.route('/api/devops/identities', methods=['GET'])
@login_required
def devops_identities():
    org   = os.getenv('AZURE_DEVOPS_ORG')
    pat   = os.getenv('AZURE_DEVOPS_PAT')
    query = request.args.get('q', '').strip()
    if not org or not pat:
        return jsonify({'success': False, 'error': 'AZURE_DEVOPS_ORG or AZURE_DEVOPS_PAT not configured'}), 400
    if not query:
        return jsonify({'success': True, 'members': []})

    import base64
    token   = base64.b64encode(f':{pat}'.encode()).decode()
    headers = {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json',
        # IdentityPicker is an older internal API surface — api-version goes in the
        # Accept header, not as a query param like every other Azure call in this
        # file. Exact value verified via DevTools against a real org.
        'Accept': 'application/json;api-version=5.0-preview.1;excludeUrls=true;enumsAsNumbers=true;msDateFormat=true;noArrayWrap=true',
    }
    url  = f'https://dev.azure.com/{org}/_apis/IdentityPicker/Identities'
    body = {
        'query': query,
        'identityTypes': ['user', 'servicePrincipal'],
        'operationScopes': ['ims', 'source'],
        'options': {'MinResults': 5, 'MaxResults': 20},
        'properties': ['DisplayName', 'SamAccountName', 'Active', 'SubjectDescriptor', 'Mail'],
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        members = []
        for result in data.get('results', []):
            for identity in result.get('identities', []):
                if identity.get('active') is False:
                    continue
                email = identity.get('mail') or identity.get('signInAddress') or identity.get('samAccountName', '')
                if email:
                    members.append({'email': email, 'name': identity.get('displayName', email)})
        return jsonify({'success': True, 'members': members})
    except requests.exceptions.HTTPError:
        return jsonify({'success': False, 'error': _safe_error(f'Azure DevOps {resp.status_code}: {resp.text}', 'devops_identities')}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'devops_identities')}), 500


@app.route("/api/agent/qa_check_ticket", methods=['POST'])
@login_required
def agent_qa_check_ticket():
    """Agent 4, read-only: look up one Azure ticket by ID, validate it's in
    AZURE_QA_STATE, recheck its url, return findings for a human to act on.
    Writes nothing — see qa_mark_done/qa_post_comment below."""
    if not AGENT4_ENABLED:
        return jsonify({'success': False, 'error': 'Agent 4 (QA) is disabled — set AGENT4_ENABLED=true to enable.'}), 503
    data = request.get_json()
    ticket_id = data.get('ticket_id')
    if not ticket_id:
        return jsonify({'success': False, 'error': 'No ticket ID provided.'}), 400
    try:
        ticket_id = int(ticket_id)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'ticket_id must be a number.'}), 400
    from src.agents.qa_agent import check_ticket
    try:
        result = check_ticket(ticket_id)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'agent_qa_check_ticket')}), 500

@app.route("/api/agent/qa_mark_done", methods=['POST'])
@login_required
def agent_qa_mark_done():
    """Human clicked 'Mark Done' for a ticket Agent 4 confirmed as resolved."""
    data = request.get_json()
    ticket_id = data.get('ticket_id')
    if not ticket_id:
        return jsonify({'success': False, 'error': 'ticket_id is required.'}), 400
    try:
        ticket_id = int(ticket_id)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'ticket_id must be a number.'}), 400
    from src.agents.qa_agent import mark_ticket_done
    try:
        ok = mark_ticket_done(ticket_id)
        return jsonify({'success': ok})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'agent_qa_mark_done')}), 500

@app.route("/api/agent/qa_post_comment", methods=['POST'])
@login_required
def agent_qa_post_comment():
    """Human clicked 'Post Comment' for a ticket still failing Agent 4's recheck."""
    data = request.get_json()
    project = data.get('project') or os.getenv('AZURE_DEVOPS_PROJECT', '')
    ticket_id = data.get('ticket_id')
    comment = data.get('comment', '').strip()
    if not project or not ticket_id or not comment:
        return jsonify({'success': False, 'error': 'project, ticket_id, and comment are required.'}), 400
    try:
        _validate_project_name(project)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    try:
        ticket_id = int(ticket_id)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'ticket_id must be a number.'}), 400
    from src.agents.qa_agent import post_qa_comment
    try:
        ok = post_qa_comment(project, ticket_id, comment)
        return jsonify({'success': ok})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'agent_qa_post_comment')}), 500


def _run_qa_bulk(ticket_ids, project):
    global _qa_bulk_state
    from src.agents.qa_agent import check_ticket, mark_ticket_done, post_qa_comment
    collected = []
    try:
        for tid in ticket_ids:
            try:
                tid_int = int(tid)
                result = check_ticket(tid_int)
                ticket_url = result.get('ticket_url', '')
                title = result.get('title', '')
                if result.get('error'):
                    collected.append({'ticket_id': tid, 'ticket_url': ticket_url, 'title': title, 'status': 'skipped', 'reason': result['error']})
                elif not result.get('still_present'):
                    mark_ticket_done(tid_int)
                    collected.append({'ticket_id': tid, 'ticket_url': ticket_url, 'title': title, 'status': 'done'})
                else:
                    fix = result.get('ai_how_to_fix')
                    comment = f"Still detected: '{result.get('issue')}'. Suggested fix:\n{fix}" if fix else f"Still detected: '{result.get('issue')}' on {result.get('url')}."
                    post_qa_comment(project, tid_int, comment)
                    collected.append({'ticket_id': tid, 'ticket_url': ticket_url, 'title': title, 'status': 'still_present'})
            except Exception as e:
                collected.append({'ticket_id': tid, 'ticket_url': '', 'title': '', 'status': 'skipped', 'reason': _safe_error(e, '_run_qa_bulk')})
    finally:
        _qa_bulk_state = {'running': False, 'results': collected}


@app.route("/api/agent/qa_tickets_by_tag", methods=['GET'])
@login_required
def qa_tickets_by_tag():
    """Fetch all Azure tickets tagged 'qa' for the bulk QA run checklist."""
    import base64
    from urllib.parse import quote as url_quote
    project = request.args.get('project') or os.getenv('AZURE_DEVOPS_PROJECT', '')
    if not project:
        return jsonify({'success': False, 'error': 'No Azure project selected.'}), 400
    try:
        _validate_project_name(project)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    org = os.getenv('AZURE_DEVOPS_ORG')
    if not org:
        return jsonify({'success': False, 'error': 'AZURE_DEVOPS_ORG not configured.'}), 500
    pat = os.getenv('AZURE_DEVOPS_PAT')
    try:
        token = base64.b64encode(f':{pat}'.encode()).decode()
        headers = {'Authorization': f'Basic {token}', 'Content-Type': 'application/json'}
        qa_state = os.getenv('AZURE_QA_STATE', 'QA')
        # Org-scoped (no project in the URL) — the Wiql REST API doesn't require project
        # in the path, so it's expressed in the WHERE clause instead (safe: project was
        # just validated above via _validate_project_name's character allowlist, which
        # forbids the quote characters a WIQL string-literal breakout would need).
        wiql_url = f'https://dev.azure.com/{org}/_apis/wit/wiql?api-version=7.1'
        # Requires BOTH the tag AND state=='QA' — a ticket carrying only one (e.g. someone
        # manually retagged or restated it outside the normal Agent 3 fix flow) shouldn't
        # surface as a ready-to-check candidate here.
        query = (f"SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = '{project}' "
                 f"AND [System.Tags] CONTAINS '{AZURE_QA_TAG}' AND [System.State] = '{qa_state}' "
                 f"ORDER BY [System.Id] DESC")
        resp = requests.post(wiql_url, headers=headers, json={"query": query}, timeout=10)
        resp.raise_for_status()
        work_items = resp.json().get('workItems', [])
        if not work_items:
            return jsonify({'success': True, 'tickets': []})
        ids = ','.join(str(w['id']) for w in work_items)
        fields = 'System.Id,System.Title,System.State,System.Tags'
        # Also org-scoped — the work item ids are already resolved above, and the
        # Work Items - List REST API doesn't require project in the path either.
        batch_url = f'https://dev.azure.com/{org}/_apis/wit/workitems?ids={ids}&fields={fields}&api-version=7.1'
        resp2 = requests.get(batch_url, headers=headers, timeout=10)
        resp2.raise_for_status()
        tickets = []
        for item in resp2.json().get('value', []):
            f = item.get('fields', {})
            tid = item['id']
            tickets.append({
                'ticket_id': tid,
                'title': f.get('System.Title', ''),
                'state': f.get('System.State', ''),
                'tags': f.get('System.Tags', ''),
                'ticket_url': f'https://dev.azure.com/{org}/{url_quote(project)}/_workitems/edit/{tid}'
            })
        return jsonify({'success': True, 'tickets': tickets})
    except Exception as e:
        return jsonify({'success': False, 'error': _safe_error(e, 'qa_tickets_by_tag')}), 500


@app.route("/api/agent/qa_bulk_run", methods=['POST'])
@login_required
def qa_bulk_run():
    """Start a background bulk QA check on the given ticket IDs (tagged with AZURE_QA_TAG).
    Auto-marks resolved tickets Done; posts a comment on still-present ones."""
    if not AGENT4_ENABLED:
        return jsonify({'success': False, 'error': 'Agent 4 (QA) is disabled — set AGENT4_ENABLED=true to enable.'}), 503
    global _qa_bulk_state
    if _qa_bulk_state.get('running'):
        return jsonify({'success': False, 'error': 'QA bulk run already in progress.'}), 409
    data = request.get_json()
    ticket_ids = data.get('ticket_ids', [])
    project = data.get('project') or os.getenv('AZURE_DEVOPS_PROJECT', '')
    if not ticket_ids:
        return jsonify({'success': False, 'error': 'No ticket IDs provided.'}), 400
    if not project:
        return jsonify({'success': False, 'error': 'No Azure project selected.'}), 400
    try:
        _validate_project_name(project)
    except ValueError as e:
        app.logger.warning("Project validation failed in qa_bulk_run: %s", e)
        return jsonify({'success': False, 'error': 'Invalid project value provided.'}), 400
    _qa_bulk_state = {'running': True, 'results': None}
    threading.Thread(target=_run_qa_bulk, args=(ticket_ids, project), daemon=True).start()
    return jsonify({'queued': True})


@app.route("/api/agent/qa_bulk_results", methods=['GET'])
@login_required
def qa_bulk_results():
    return jsonify({'ready': not _qa_bulk_state.get('running', False), 'results': _qa_bulk_state.get('results')})


@app.route("/api/agent/token_usage", methods=['GET'])
@login_required
def agent_token_usage():
    """Token usage for one panel's worth of agents: aggregate totals (per agent/model)
    plus the scrollable per-call log, newest first. ?agent=agent2,agent3 scopes both
    (comma-separated for a panel shared by more than one agent); omit for everything."""
    agent_param = request.args.get('agent')
    agent = agent_param.split(',') if agent_param else None
    return jsonify({
        'success': True,
        'summary': get_usage_summary(agent),
        'log': get_usage_log(agent),
    })


@app.route("/api/agent/provider_options", methods=['GET'])
def provider_options():
    return jsonify({
        'anthropic_available': bool(os.getenv('ANTHROPIC_API_KEY')),
        'openai_available': bool(os.getenv('OPENAI_API_KEY')),
        'current': get_provider()
    })

@app.route("/api/agent/set_provider", methods=['POST'])
def set_provider():
    data = request.get_json()
    provider = data.get('provider')
    if provider not in ('anthropic', 'openai'):
        return jsonify({'success': False, 'error': 'Invalid provider'}), 400
    if provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
        return jsonify({'success': False, 'error': 'Anthropic API key not configured'}), 400
    if provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
        return jsonify({'success': False, 'error': 'OpenAI API key not configured'}), 400
    set_provider_override(provider)
    return jsonify({'success': True, 'provider': provider})

@app.route("/api/agent/chat", methods=['POST'])
def agent_chat():
    provider = get_provider()
    if provider is None:
        return jsonify({'success': False, 'error': 'No AI provider configured'}), 400

    data    = request.get_json()
    message = data.get('message', '')
    issues  = data.get('issues', [])

    issue_lines = '\n'.join(
        f"{iss.get('issue','')} | {iss.get('url','')} | priority:{iss.get('priority','?')} | type:{iss.get('type','')}"
        for iss in issues
    )

    # Prompt built in src/agents/prompts.py — see build_agent_chat_prompt()
    prompt = build_agent_chat_prompt(issue_lines, message)

    try:
        if provider == 'anthropic':
            resp = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                system=AGENT_CHAT_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=4096,
            )
            text = resp.content[0].text
            input_tokens, output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            resp = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {'role': 'system', 'content': AGENT_CHAT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=4096,
            )
            text = resp.choices[0].message.content
            input_tokens, output_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens

        record_usage('agent2', ANTHROPIC_MODEL if provider == 'anthropic' else OPENAI_MODEL, input_tokens, output_tokens, label='agent_chat')

        json_match = re.search(r'\{\s*"matches"\s*:', text)
        if json_match:
            json_start = json_match.start()
            json_end   = text.rfind('}') + 1
            match_data = json.loads(text[json_start:json_end])
            reply      = text[:json_start].strip()
        else:
            match_data = {'matches': []}
            reply      = text.strip()

        return jsonify({
            'success': True,
            'reply': reply,
            'matches': match_data.get('matches', [])
        })

    except Exception as e:
        print(f"[Chat] Error: {e}")
        return jsonify({'success': False, 'error': 'Chat request failed'}), 500

def main():
    import signal

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Recover any crashed crawls from previous session
    recover_crashed_crawls()

    # Start cleanup thread for old crawler instances
    start_cleanup_thread()

    print("=" * 60)
    print("LibreCrawl - SEO Spider")
    print("=" * 60)
    print(f"\n🚀 Server starting on http://0.0.0.0:5000")
    print(f"🌐 Access from browser: http://localhost:5000")
    print(f"📱 Access from network: http://<your-ip>:5000")
    print(f"\n✨ Multi-tenancy enabled - each browser session is isolated")
    print(f"💾 Settings stored in browser localStorage")
    print(f"\nPress Ctrl+C to stop the server\n")
    print("=" * 60 + "\n")

    # Open browser in a separate thread after short delay
    def open_browser():
        time.sleep(1.5)  # Wait for Flask to start
        webbrowser.open('http://localhost:5000')

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Run Flask server with Waitress (production-grade WSGI server)
    from waitress import serve
    print("Starting LibreCrawl on http://localhost:5000")
    print("Using Waitress WSGI server with multi-threading support")
    serve(app, host='0.0.0.0', port=5000, threads=8)

if __name__ == '__main__':
    main()