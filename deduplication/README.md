# Deduplication

A tool for checking code duplicates. Taken from: https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/codeutils/deduplication, see also: https://arxiv.org/abs/1812.06469.

Usage:

```bash
deduplication.py --data_dir <data_dir> --output_dir <output_dir>
```

`data_dir` should be a directory containing .zst compressed files. Each file should be in the `jsonl` format. Each `jsonl` entry should have a `text` field with the code as a string, and a `meta` field which is a dictionary containing `repo_name` and `file_name`. `output_dir` will be the directory containing the deduplicated data, in the same format as the input data.

The deduplication tool will load each file, tokenize the code, and then count the occurrences of each identifier within the code. The identifiers are obtained by regexing out all the non-alphanumeric tokens. Once the identifier counts are obtained for each code example in the data, we measure the similarity between every pair of identifier counts. Similarities over a threshold are marked as duplicates and are used to create a duplicate cluster. We then calculate which documents to exclude by removing one document from each cluster to keep, and leaving the rest as documents to exclude. Finally, we load the files again, checking if each one is a duplicate and writing the non-duplicate ones to `output_dir` in the same format as the input `data_dir`.

The tool creates two files:

- `duplicate_clusters.txt`: all of the duplicate clusters. Each line is a cluster and are formatted as: `index|repo_name|file_name|code_hash,index|repo_name|file_name|code_hash`.

- `documents_to_exclude.txt`: the documents to remove, i.e. all but one item from each duplicate cluster. Each new line is a document to remove and are formatted as: `index|repo_name|file_name|code_hash`.
