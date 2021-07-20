import os

import pandas as pd

from fastcore.script import *
from ghapi.all import GhApi

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


# Open issue on repo using custom title and body
def get_license_info(owner, repo):
    api = GhApi(owner=owner, repo=repo, token=GITHUB_TOKEN)
    license = api.licenses.get_for_repo(owner=owner, repo=repo)
    return license.license.name

@call_parse
def main(repos_path: Param("Path to the csv containing all of the repos", str)):
    """
    Use pandas dataframe from the repos path to open issues in each of them.
    """
    repos_path = Path(repos_path)
    df = pd.read_csv(repos_path)

    # Loop through repos and get their license
    licenses = []
    for _, row in df.iterrows():
        owner, repo = row["name"].split("/")
        licenses.append(get_license_info(owner, repo))
    df["license"] = licenses
    df.to_csv(repos_path.parent/f"{repos_path.stem}_with_license.csv", index=False)
