from metrics.bleu import compute_bleu


def compute_exact_match(references,generated)->float:
    """
    Computes Exact Match Accuracy.
    args:
        reference: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
        translation: list of translations to score. Each translation
          should be tokenized into a list of tokens.
    returns:
        exact_match_accuracy : Float
    """
    assert(len(references[0])==len(generated),"Number of Samples should be equal in References and Synthesized Outputs..")
    exact_match_count = 0.0
    for gen,ref in zip(generated, references[0]):
        if gen == ref:
            exact_match_count += 1
    exact_match_acc = exact_match_count/len(generated)
    return exact_match_acc

def compute_metrics(references,generated) -> dict:
    """
    Calculates various metrics and returns the calculated dict of these matrics.
    args:
        reference: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
        translation: list of translations to score. Each translation
          should be tokenized into a list of tokens.
    returns:
        A dicitonary with different metrics intact.
    """
    metrics_dict = {} #Update as in new metrics are added over here.
    metrics_dict["smoothed_bleu_4"] = compute_bleu(references,generated,smooth=True)
    metrics_dict["bleu_4"] = compute_bleu(references,generated,smooth=False)
    metrics_dict["exact_match_acc"] = compute_exact_match(references,generated)
    return metrics_dict