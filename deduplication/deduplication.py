import argparse
from duplicate_detector import DocumentID, DuplicateDetector
import os
import lm_dataformat
import tqdm

parser = argparse.ArgumentParser(description="Deduplicate a list of files")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--set_similarity_threshold", type=float, default=0.8)
parser.add_argument("--multiset_similarity_threshold", type=float, default=0.7)
parser.add_argument("--min_num_tokens_per_document", type=int, default=20)
args = parser.parse_args()

data_files = [
    os.path.join(args.data_dir, data_file) for data_file in os.listdir(args.data_dir)
]

duplicate_detector = DuplicateDetector(
    args.set_similarity_threshold,
    args.multiset_similarity_threshold,
    args.min_num_tokens_per_document,
)

rdr = lm_dataformat.Reader(data_files)

for i, doc in enumerate(
    tqdm.tqdm(rdr.stream_data(get_meta=True), desc="adding files to duplicate detector")
):
    code, metadata = doc
    document_id = DocumentID(
        index=i,
        repo_name=metadata["repo_name"],
        file_name=metadata["file_name"],
    )
    duplicate_detector.add_file(document_id, code)

duplicate_clusters = duplicate_detector.get_duplicate_clusters()
duplicate_detector.print_duplicate_clusters_stats(duplicate_clusters)

with open("duplicate_clusters.txt", "w+") as f:
    for duplicate_cluster in duplicate_clusters:
        cluster_str = ",".join(
            [
                f"{document_id.index}|{document_id.repo_name}|{document_id.file_name}"
                for document_id in duplicate_cluster
            ]
        )
        f.write(f"{cluster_str}\n")

with open("documents_to_exclude.txt", "w+") as f:
    document_ids_to_exclude = duplicate_detector.get_documents_to_exclude(
        duplicate_clusters
    )
    for document_id in document_ids_to_exclude:
        f.write(
            f"{document_id.index}|{document_id.repo_name}|{document_id.file_name}\n"
        )
