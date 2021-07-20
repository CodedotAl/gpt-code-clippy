# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# The following code is taken from CodeXGlue Repository - https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/evaluator/CodeBLEU/bleu.py


"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (
                possible_matches_by_order[i] + 1.0
            )
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (
                    float(matches_by_order[i]) / possible_matches_by_order[i]
                )
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)
    bleu = geo_mean * bp
    bleu_score_dict = {
        "bleu": bleu,
        "precision": precisions,
        "bp": bp,
        "ratio": ratio,
        "trans_len": translation_length,
        "ref_len": reference_length,
    }
    return bleu_score_dict  # (bleu, precisions, bp, ratio, translation_length, reference_length)


def bleu_test_case():
    """A simple functionality test case to evaluate BLEU"""
    generated = [[["a", "=", "b", "\n", "y", "=", "a", "+", "1"]]]
    reference = [["a", "=", "b", "\n", "print", "a"]]
    score_dict = compute_bleu(generated, reference, smooth=False)
    return score_dict


if __name__ == "__main__":
    score_dict = bleu_test_case()
    print(score_dict)
