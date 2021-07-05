import os

from ghapi.all import GhApi

ISSUE_TITLE = "Participation in an Open Source Language Modeling Dataset"

api = GhApi()
issues = api.issues.list_for_repo(
    owner="ncoop57", repo="gpt-code-clippy", state="all", creator="ncoop57"
)

# Loop through each issue and check if it contains the title
ISSUE_ID = -1
for i in issues:
    if i.title == ISSUE_TITLE:
        ISSUE_ID = i.number
        print(i.title)
        print("Found it!")
        break

comments = api.issues.list_comments_for_repo(
    owner="ncoop57",
    repo="gpt-code-clippy",
    state="all",
)

# Loop through each comment and check if it is a reply to the issue
for c in comments:
    idx = c.issue_url.split("/")[-1]
    if int(idx) == ISSUE_ID:
        print(int(idx))
        print(c.body)
