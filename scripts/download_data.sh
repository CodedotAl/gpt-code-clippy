#! /bin/bash
REPO_SHARDS=$1
OUT_DIR=$2
ID=0
for shard in ${REPO_SHARDS}/*; do
    echo "Processing data from ${shard}"
    python convert_to_gh_downloader_format.py $shard $OUT_DIR
    ID=$(($ID + 1))
    cd $OUT_DIR
    python download_repo_text.py
    mv $OUT_DIR"github_data" $OUT_DIR"github_data_${ID}"
    cd -
    echo "Finished processsing data from ${shard}"
done
# python convert_to_gh_downloader_format.py $COMBINED_REPOS $OUT_DIR

# cd $OUT_DIR
# python download_repo_text.py
# cd -