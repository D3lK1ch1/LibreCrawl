from mcp.server.mcpserver import MCPServer
import os
import requests
from dotenv import load_dotenv
load_dotenv()

http_session = requests.Session()

BASE_URL = os.getenv('MCP_BASE_URL', 'http://localhost:5000')
MCP_SERVICE_USERNAME = os.getenv('MCP_SERVICE_USERNAME', '')
MCP_SERVICE_PASSWORD = os.getenv('MCP_SERVICE_PASSWORD', '')
mcp = MCPServer("LibreCrawl")


def _login():
    """Authenticate http_session as the dedicated service account. Under LOCAL_MODE
    this is never needed — every call already succeeds via LOCAL_MODE's auto-login —
    so this only does anything once a call actually comes back unauthenticated."""
    if not MCP_SERVICE_USERNAME or not MCP_SERVICE_PASSWORD:
        return False
    try:
        resp = http_session.post(f"{BASE_URL}/api/login", json={
            "username": MCP_SERVICE_USERNAME,
            "password": MCP_SERVICE_PASSWORD,
        })
        return resp.ok and resp.json().get('success', False)
    except Exception as e:
        print(f"[MCP] Service-account login failed: {e}")
        return False


def _request(method, url, **kwargs):
    """requests.Session wrapper: on a 401/403, logs in the service account once and
    retries — covers both a fresh process (never authenticated yet) and a session
    that went stale (e.g. expired after an hour of inactivity, per README)."""
    resp = http_session.request(method, url, **kwargs)
    if resp.status_code in (401, 403) and _login():
        resp = http_session.request(method, url, **kwargs)
    return resp

@mcp.tool()
def start_crawl(url: str) -> dict:
    """Starts a crawl for the given URL. Call poll_crawl_status after this until the crawl is complete."""
    response = _request('POST', f"{BASE_URL}/api/start_crawl", json={"url": url})
    return response.json()

@mcp.tool()
def poll_crawl_status() -> dict:
    """Polls the status of an ongoing crawl. Returns status, urls[], issues[], and stats. Keep calling until status equals completed."""
    response = _request('GET', f"{BASE_URL}/api/crawl_status")
    return response.json()

@mcp.tool()
def explain_issue(url: str, issue: str, category: str, details: str) -> dict:
    """Given an issue ID from the crawl results, returns a detailed explanation of the issue and how to fix it."""
    response = _request('POST', f"{BASE_URL}/api/explain_issue", json={
        "url": url,
        "issue": issue,
        "category": category,
        "details": details
    })
    return response.json()

@mcp.tool()
def check_existing_tickets(url: str, issue: str, category: str, details: str) -> dict:
    """Checks if there are existing tickets for the given issue ID. Returns a list of tickets if they exist."""
    response = _request('POST', f"{BASE_URL}/api/devops_tickets/check", json={
        "pairs": [{
            "url": url,
            "issue": issue,}]
    })
    return response.json()

@mcp.tool()
def create_devops_ticket(url: str, issue: str, category: str, issue_type: str, ai_explanation: str, ai_how_to_fix: str, ai_priority: str, ai_role: str) -> dict:
    """Creates a new Azure DevOps ticket for the given issue ID with the provided title and description."""
    response = _request('POST', f"{BASE_URL}/api/create_devops_ticket", json={
        "url": url,
        "issue": issue,
        "category": category,
        "issue_type": issue_type,
        "ai_explanation": ai_explanation,
        "ai_how_to_fix": ai_how_to_fix,
        "ai_priority": ai_priority,
        "ai_role": ai_role
    })
    return response.json()

@mcp.tool()
def create_bulk_tickets(approved: list[dict], project: str = "", user_id: str | None = None):
    """
    Create multiple tickets from approved crawl issues.
    """
    response = _request('POST', f"{BASE_URL}/api/agent/create_bulk_tickets", json={
        "approved" : approved,
    })
    response.raise_for_status()
    return response.json()


@mcp.tool()
def probe_urls(issues: list) -> dict:
    """Probes a list of URLs to check their HTTP status codes and returns the results."""
    response = _request('POST', f"{BASE_URL}/api/probe_http_errors", json={"issues" : issues})
    return response.json()

@mcp.tool()
def poll_workflow_trigger() -> dict:
    """Polls the status of an ongoing workflow trigger. Returns status and details. Keep calling until status equals completed."""
    response = _request('GET', f"{BASE_URL}/api/agent/workflow_trigger")
    return response.json()

@mcp.tool()
def post_review_results(issues: list) -> dict:
    """Posts the reviewd issue list to the browser for human review."""
    response = _request('POST', f"{BASE_URL}/api/agent/review", json={
        "issues": issues,
    })
    return response.json()

@mcp.tool()
def poll_approval() -> dict:
    """Polls for approval status of a given issue. Returns status and details. Keep calling until status equals approved or rejected."""
    response = _request('GET', f"{BASE_URL}/api/agent/approval")
    return response.json()

if __name__ == "__main__":
    mcp.run()