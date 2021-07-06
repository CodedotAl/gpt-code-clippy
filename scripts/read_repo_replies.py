import os

from ghapi.all import GhApi

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ISSUE_TITLE = "Participation in an Open Source Language Modeling Dataset"
REPOS = [
    "ProgrammingSpace/test-repo-3",
    "ncoop57/deep_parking",
    "ncoop57/test-repo",
    "ncoop57/recipe_name_suggester",
    "ncoop57/test-repo-2",
]

for r in REPOS:
    print(r)
    owner, repo = r.split("/")
    api = GhApi(owner=owner, repo=repo, token=GITHUB_TOKEN)
    issues = api.issues.list_for_repo(owner=owner, repo=repo, state="all")
    for i in issues:
        if i.title == ISSUE_TITLE:
            if i.reactions["-1"] > 0:
                print(f"{r} is opting out")
                api.issues.create_comment(
                    i.number, body="Thank you, your repository has been removed."
                )
                api.issues.update(i.number, state="closed")
