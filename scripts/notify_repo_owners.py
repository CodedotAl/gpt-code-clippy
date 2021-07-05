import os

from ghapi.all import GhApi

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ISSUE_TITLE = "Participation in an Open Source Language Modeling Dataset"
ISSUE_BODY = """Hi there, your repository has been selected to be included in an effort
to train an open source version of GitHub and OpenAI's [Copilot tool](https://copilot.github.com/).
You can find more information on our project [here](https://github.com/ncoop57/gpt-code-clippy).

If you are the owner/admin of this repository and would like to opt-out of this,
please reply to this issue before July 9th with "yes" and we will remove your
repository from our list.
"""

api = GhApi(owner="ncoop57", repo="gpt-code-clippy", token=GITHUB_TOKEN)
issue = api.issues.create(title=ISSUE_TITLE, body=ISSUE_BODY)
# api.issues.update(issue.number, state="closed")


# print(os.environ["GITHUB_TOKEN"])
