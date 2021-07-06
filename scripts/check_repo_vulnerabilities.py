import pandas as pd

from fastcore.script import *
from joblib import Parallel, delayed
from pathlib import Path
from subprocess import CalledProcessError, check_output

SCORECARD_PATH = Path("../scripts/scorecard")


def _run_scorecard(repo):
    # get results of scorecard
    results = check_output(
        [
            SCORECARD_PATH,
            f"--repo=github.com/{repo}",
            "--checks=Vulnerabilities",
            "--format=csv",
        ]
    ).decode("utf-8")

    # parse results and convert to propert data types
    results = results.split("\n")[1]
    repo, no_vulns, confidence = results.split(",")
    no_vulns = True if no_vulns == "true" else False
    confidence = int(confidence)

    # check results has max confidence.
    # If not return None else return whether it does not have a vulnerability
    if confidence < 10:
        return None
    else:
        return no_vulns


@call_parse
def run_scorecard(
    repos_path: Param("Path to the csv containing all of the repos", str)
):
    repos_path = Path(repos_path)
    df = pd.read_csv(repos_path).head()

    # Loop through each repo parrallely and call the scorecard script
    # TODO: Need to let sleep so GitHub api limit is not exceeded
    vulnerabilities = Parallel(n_jobs=-1)(
        delayed(_run_scorecard)(repo) for repo in df["name"]
    )
    df["no_vulnerability"] = vulnerabilities

    # Save the results to a csv
    df.to_csv(repos_path.parent / "combined_repos.csv", index=False)
