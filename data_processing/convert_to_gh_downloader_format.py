import numpy as np
import pandas as pd

from fastcore.script import *
from pathlib import Path


def convert_to_gh_downloader(df, path):
    # select only name, stargazers, and languages columns and save to path
    df = df[["name", "stargazers", "languages"]]
    df.to_csv(path / "github_repositories.csv", index=False, header=False)


@call_parse
def main(
    repos_path: Param("Path to the csv containing all of the repos", str),
    output_path: Param("Path to the directory where the output will be saved", str),
):
    repos_path = Path(repos_path)
    output_path = Path(output_path)
    df = pd.read_csv(repos_path)
    convert_to_gh_downloader(df, output_path)
