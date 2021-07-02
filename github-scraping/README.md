# Github Scraping Scripts

This directory contains scripts to scrape github (based on scripts originally by [Vincent Hellendoorn](https://vhellendoorn.github.io/)).

* `01_metadata_by_github_api.py`: A script to crawl repo metadata using the github API. Currently it crawls N repositories of down to M github stars for language L.
* `02_clone_repos.sh`: Actually download the repos.
* `03_lexer.py`: Finds and lexes code by file extension. Note that this does some stuff that might be bad for model training such as getting rid of whitespace, so it might need to be fixed.
