import os

import pandas as pd

from fastcore.script import *
from ghapi.all import GhApi

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ISSUE_TITLE = "Participation in an Open Source Language Modeling Dataset"
ISSUE_BODY = """Hi there, your repository has been selected to be included in an effort
to train an open source version of GitHub and OpenAI's [Copilot tool](https://copilot.github.com/).
You can find more information on our project [here](https://github.com/ncoop57/gpt-code-clippy).

If you are the owner/admin of this repository and would like to opt-out of this,
please downvote this issue before July 9th and we will remove your repository
from our list. We will comment an acknowledgement in this issue if you choose
to opt-out on July 9th. If you have any questions, please open an issue on our
[repository](https://github.com/ncoop57/gpt-code-clippy/issues/new).
"""

REPOS = [
    "ProgrammingSpace/test-repo-3",
    "ncoop57/deep_parking",
    "ncoop57/test-repo",
    "ncoop57/recipe_name_suggester",
    "ncoop57/test-repo-2",
]

# Open issue on repo using custom title and body
def open_issue(owner, repo):
    api = GhApi(owner=owner, repo=repo, token=GITHUB_TOKEN)
    api.issues.create(title=ISSUE_TITLE, body=ISSUE_BODY)


@call_parse
def main(repos_path: Param("Path to the csv containing all of the repos", str)):
    """
    Use pandas dataframe from the repos path to open issues in each of them.
    """
    df = pd.read_csv(repos_path)

    # Loop through repos and open issue for each repo
    for _, row in df.iterrows():
        owner, repo = row["name"].split("/")
        open_issue(owner=owner, repo=repo)
