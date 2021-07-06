#! /bin/bash
COMBINED_REPOS=$1
OUT_DIR=$2
python convert_to_gh_downloader_format.py $COMBINED_REPOS $OUT_DIR

cd $OUT_DIR
python download_repo_text.py
cd -