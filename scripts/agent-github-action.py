import json
import os
import sys
import urllib.error
import urllib.request

# GitHub tokens never contain whitespace; strip all of it, not just leading/
# trailing, since a copy-paste can leave a stray character anywhere in the
# secret and Python's http.client rejects any of it in a header value.
agent_github_token = "".join(os.environ["AGENT_GITHUB_TOKEN"].split())

payload = json.loads(os.environ["AGENT_REQUEST"])
allowed = {"agent", "operation", "repo", "number", "title", "body", "head", "base", "reaction"}
unknown = set(payload) - allowed
if unknown:
    raise SystemExit(f"Champs interdits: {sorted(unknown)}")

if payload.get("agent") != os.environ["EXPECTED_AGENT"]:
    raise SystemExit("Identité d'agent refusée")

repo = payload.get("repo", "")
if repo != os.environ["GITHUB_REPOSITORY"]:
    raise SystemExit("Dépôt refusé")

operation = payload.get("operation")
number = payload.get("number")

if operation == "issue.create":
    method, path = "POST", f"/repos/{repo}/issues"
    data = {"title": payload["title"], "body": payload.get("body", "")}
elif operation == "comment.create":
    method, path = "POST", f"/repos/{repo}/issues/{int(number)}/comments"
    data = {"body": payload["body"]}
elif operation == "reaction.create":
    allowed_reactions = {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
    reaction = payload.get("reaction")
    if reaction not in allowed_reactions:
        raise SystemExit("Réaction refusée")
    method, path = "POST", f"/repos/{repo}/issues/{int(number)}/reactions"
    data = {"content": reaction}
elif operation == "pr.create":
    method, path = "POST", f"/repos/{repo}/pulls"
    data = {
        "title": payload["title"],
        "body": payload.get("body", ""),
        "head": payload["head"],
        "base": payload.get("base", "main"),
    }
else:
    raise SystemExit("Opération refusée")

request = urllib.request.Request(
    "https://api.github.com" + path,
    data=json.dumps(data).encode(),
    method=method,
    headers={
        "Authorization": "Bearer " + agent_github_token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "claude-agent-github-bridge",
    },
)

try:
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode(), file=sys.stderr)
    raise

print(json.dumps({"url": result.get("html_url"), "id": result.get("id")}, indent=2))
