# Deduplication

A tool for checking code duplicates by assuming two files with the same sequence of variables are duplicates.

Usage:

```bash
deduplication.py --data_dir <data_dir> --output_dir <output_dir>
```

`data_dir` should be a directory containing .zst compressed files. Each file should be in the `jsonl` format. Each `jsonl` entry should have a `text` field with the code as a string, and a `meta` field which is a dictionary containing `repo_name` and `file_name`. `output_dir` will be the directory containing the deduplicated data, in the same format as the input data.

The deduplication tool will load each file, tokenize the code, obtaining a list of variables within the code. The variables are obtained by regexing out all tokens which are not made up of only alphanumeric characters. We then get the list of unique variable sequences, filter our dataset so we only have one of each sequence, and write the filtered dataset to `output_dir` in the same format as the input data in `data_dir`.
