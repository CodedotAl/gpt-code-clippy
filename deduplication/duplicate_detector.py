import collections
import re
from SetSimilaritySearch import all_pairs
import numpy as np
import tqdm


class DocumentID:
    def __init__(self, index, repo_name, file_name):
        self.index = index
        self.repo_name = repo_name
        self.file_name = file_name


class DuplicateDetector:
    def __init__(
        self,
        set_similarity_threshold: float,
        multiset_similarity_threshold: float,
        min_num_tokens_per_document: int,
    ):
        self.vocabulary = {}
        self.set_similarity_threshold = set_similarity_threshold
        self.multiset_similarity_threshold = multiset_similarity_threshold
        self.min_num_tokens_per_document = min_num_tokens_per_document
        self.document_ids = []
        self.document_elements = []

    def get_token_id(self, token: str) -> int:
        """Converts a token into an integer representation"""
        # get the token id from vocabulary
        token_id = self.vocabulary.get(token)
        # if the token is not in the vocabulary, assign it a new id and add it to the vocabulary
        if token_id is None:
            token_id = len(self.vocabulary)
            self.vocabulary[token] = token_id
        return token_id

    def add_file(self, document_id: DocumentID, code: str, language=None) -> bool:
        """Add a file to the documents to be viewed by the duplicate detector"""
        # split on non-alphanumeric characters
        tokens = re.split(r"\W+", code)
        # skip documents with too few tokens
        if len(tokens) < self.min_num_tokens_per_document:
            return False
        # get token ids
        ids = [self.get_token_id(token) for token in tokens]
        # count the number of occurences of each id
        id_counter = collections.Counter(ids)
        self.document_ids.append(document_id)
        self.document_elements.append(id_counter)
        return True

    def get_duplicate_pairs(self):
        """Find the pairs of documents that are duplicates."""
        # get all similar pairs of documents
        similar_pairs = all_pairs(
            self.document_elements,
            similarity_func_name="jaccard",
            similarity_threshold=self.set_similarity_threshold,
        )
        for index_1, index_2, _ in tqdm.tqdm(
            similar_pairs, desc="computing duplicates..."
        ):
            if (
                self.get_multiset_jaccard_similarity(index_1, index_2)
                >= self.multiset_similarity_threshold
            ):
                yield index_1, index_2

    def get_multiset_jaccard_similarity(self, index_1: int, index_2: int) -> float:
        """Calculate the multiset Jaccard similarity between two documents."""
        intersection_size = sum(
            (self.document_elements[index_1] & self.document_elements[index_2]).values()
        )
        union_size = sum(
            (self.document_elements[index_1] | self.document_elements[index_2]).values()
        )
        return float(intersection_size) / union_size

    def get_duplicate_clusters(self):
        """Compute the duplicates in the indexed documents."""

        # stores duplicate clusters, list of set of DocumentID
        duplicate_clusters = []

        # get the duplicate pairs
        pairwise_relationships = collections.defaultdict(list)
        for index_1, index_2 in self.get_duplicate_pairs():
            assert index_1 != index_2
            pairwise_relationships[index_1].append(index_2)
            pairwise_relationships[index_2].append(index_1)

        # set of which documents have duplicates
        documents_with_duplicates = set(pairwise_relationships.keys())

        pbar = tqdm.tqdm(desc="creating duplicate clusters...")

        while len(documents_with_duplicates) > 0:

            # look at a document with duplicates
            current_index = documents_with_duplicates.pop()
            current_index_clones = {current_index}

            # need to visit all duplicates of this document
            documents_to_visit = pairwise_relationships[current_index]

            while len(documents_to_visit) > 0:

                # look at a document that is a duplicate of the current document
                other_index = documents_to_visit.pop()
                # add to list of clones of current document
                current_index_clones.add(other_index)
                # remove it so we don't visit the other document again
                documents_with_duplicates.discard(other_index)

                # add all of the duplicates of the other document to the list of duplicates of the current document
                documents_to_visit.extend(
                    next_index
                    for next_index in pairwise_relationships[other_index]
                    if next_index in documents_with_duplicates
                )

            # add all the duplicates of current document to the list of duplicate clusters
            duplicate_clusters.append(
                set(self.document_ids[i] for i in current_index_clones)
            )

            pbar.update(1)

        pbar.close()

        return duplicate_clusters

    def print_duplicate_clusters_stats(self, duplicate_clusters):
        total_num_files = len(self.document_ids)
        num_cloned_files = sum(len(c) for c in duplicate_clusters)
        print(f"duplicated files: {num_cloned_files / total_num_files * 100}%")
        print(f"mean files per clone {np.mean([len(c) for c in duplicate_clusters])}")
        print(
            f"median files per clone {np.median([len(c) for c in duplicate_clusters])}"
        )
        print(
            f"duplication ratio: {((num_cloned_files - len(duplicate_clusters)) / total_num_files * 100)}"
        )

    def get_documents_to_exclude(self, duplicate_clusters):
        """A set of DocumentIDs to exclude because they are duplicates."""

        # remove one document from each duplicate cluster to keep
        for cluster in duplicate_clusters:
            cluster.pop()  # remove one at random to keep

        return set.union(*duplicate_clusters)
