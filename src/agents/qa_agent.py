"""
Agent 4 — QA / Validation.
Triggered by: supervisor selecting tickets in the QA bulk panel (tag: AZURE_QA_TAG),
or clicking "Run QA" on a single ticket.
Input: Azure ticket ID. Output: pass (ticket marked Done) or fail (AI-drafted comment
posted explaining what still needs fixing).
Read-only — re-crawls the URL but writes nothing to the site. Azure writes
(mark Done, post comment) are handled by the calling route in main.py, not here.
"""
import os
import re
import base64
import requests

from src.crawler import check_single_url
from src.agents.fix_agent import set_ticket_state, add_ticket_comment

# Reuses MCP_BASE_URL — src/mcp_server.py already owns this env var for the
# identical self-referencing-local-API use case.
BASE_URL = os.getenv('MCP_BASE_URL', 'http://localhost:5000')


def _auth_headers():
    pat = os.getenv('AZURE_DEVOPS_PAT')
    token = base64.b64encode(f':{pat}'.encode()).decode()
    return {'Authorization': f'Basic {token}', 'Content-Type': 'application/json'}


def _parse_ticket_title(title):
    """Recover (category, issue) from a title shaped like '[Category] Issue — shortURL'
    (the exact format _create_single_ticket() writes, main.py). Returns None if the
    title doesn't match — i.e. this isn't a LibreCrawl-created ticket."""
    if not title or not title.startswith('['):
        return None
    try:
        category, rest = title[1:].split('] ', 1)
        issue, _short_url = rest.rsplit(' — ', 1)
        return category, issue
    except ValueError:
        return None


def _parse_ticket_url(description):
    """Recover the original URL from the Description's first anchor tag (the exact
    format _create_single_ticket() writes: '<a href="{url}">{url}</a>')."""
    if not description:
        return None
    match = re.search(r'<a href="([^"]+)">', description)
    return match.group(1) if match else None


def fetch_ticket(org, ticket_id):
    """GET a single work item by ID with just the fields the QA check needs.
    """
    headers = _auth_headers()
    fields = 'System.Title,System.Description,System.State,System.Tags,Microsoft.VSTS.Common.AcceptanceCriteria'
    url = f'https://dev.azure.com/{org}/_apis/wit/workitems/{int(ticket_id)}?fields={fields}&api-version=7.1'
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _draft_fix_suggestion(url, issue, category):
    """Drafts a suggested fix via the existing AI-explain route (the same one Agent 2
    already uses) for a still-present issue. Never raises — returns None if no AI
    provider is configured, the call fails, or this runs outside LOCAL_MODE (the route
    is @login_required and this is a server-to-itself call with no session cookie,
    same constraint mcp_server.py's explain_issue() tool already has)."""
    try:
        resp = requests.post(f'{BASE_URL}/api/explain_issue', json={
            'url': url, 'issue': issue, 'category': category, 'details': '',
        }, timeout=15)
        data = resp.json()
        if data.get('success') and data.get('how_to_fix'):
            return data['how_to_fix']
    except Exception:
        pass
    return None


def check_ticket(ticket_id):
    """Agent 4, read-only: fetch one ticket, validate it's in AZURE_QA_STATE AND carries
    the AZURE_QA_TAG tag, recover its url/issue from Title/Description, re-crawl the url,
    and return findings for a human to act on. Drafts an AI-suggested fix if the issue is
    still present. Writes nothing to Azure — see mark_ticket_done()/post_qa_comment() for
    that.
    Both the state and tag are required, not either/or — a ticket that only has one (e.g.
    a human manually moved it to 'QA' without the tag, or the tag survived a state change
    away from 'QA') isn't reliably a ticket Agent 3 actually fixed and queued for QA, so
    it's rejected just the same as one that's outright missing both.
    Returns a dict; always has ticket_id/ticket_url; 'error' is set (others may be
    missing) when the ticket can't be checked further — state_ok tells you whether
    it at least cleared the state+tag gate.
    """
    org = os.getenv('AZURE_DEVOPS_ORG')
    qa_state = os.getenv('AZURE_QA_STATE', 'QA')
    qa_tag = os.getenv('AZURE_QA_TAG', 'qa-agent')
    ticket_url = f'https://dev.azure.com/{org}/_workitems/edit/{ticket_id}'

    try:
        item = fetch_ticket(org, ticket_id)
    except Exception as e:
        return {'ticket_id': ticket_id, 'ticket_url': ticket_url, 'state_ok': False,
                'error': f'Could not fetch ticket #{ticket_id}: {e}'}

    fields = item.get('fields', {})
    state = fields.get('System.State', '')
    tags = [t.strip() for t in fields.get('System.Tags', '').split(';') if t.strip()]
    title = fields.get('System.Title', '')
    description = fields.get('System.Description', '')
    acceptance_criteria = fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '')

    base = {'ticket_id': ticket_id, 'ticket_url': ticket_url, 'title': title, 'state': state,
            'acceptance_criteria': acceptance_criteria}

    if state != qa_state or qa_tag not in tags:
        problems = []
        if state != qa_state:
            problems.append(f"state is '{state}', not '{qa_state}'")
        if qa_tag not in tags:
            problems.append(f"missing the '{qa_tag}' tag")
        return {**base, 'state_ok': False,
                'error': f"Ticket #{ticket_id} isn't ready for QA — {' and '.join(problems)}."}

    parsed_title = _parse_ticket_title(title)
    url = _parse_ticket_url(description)
    if not parsed_title or not url:
        return {**base, 'state_ok': True,
                'error': "Title/Description don't match a LibreCrawl-created ticket — can't recover the URL/issue automatically."}

    category, issue = parsed_title
    base.update({'category': category, 'issue': issue, 'url': url})

    try:
        _, detected_issues = check_single_url(url)
    except Exception as e:
        return {**base, 'state_ok': True, 'error': f'Could not re-check {url}: {e}'}

    still_present = any(i.get('issue') == issue for i in detected_issues)
    result = {**base, 'state_ok': True, 'still_present': still_present,
              'ai_how_to_fix': None, 'error': None}
    if still_present:
        result['ai_how_to_fix'] = _draft_fix_suggestion(url, issue, category)
    return result


def mark_ticket_done(ticket_id):
    """Human clicked 'Mark Done' — transitions the ticket to AZURE_QA_DONE_STATE."""
    done_state = os.getenv('AZURE_QA_DONE_STATE', 'Done')
    return set_ticket_state(ticket_id, done_state)


def post_qa_comment(project, ticket_id, comment):
    """Human clicked 'Post Comment' (with whatever text they confirmed/edited)."""
    return add_ticket_comment(ticket_id, project, comment)
