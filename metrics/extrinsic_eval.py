from metrics.bleu import compute_bleu
from metrics.parse_check import check_parse

Parser = check_parse() #Initializing parser

def compute_metrics(references,generated,lang) -> dict:
    """
    Calculates various metrics and returns the calculated dict of these matrics.
    args:
        reference: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
        translation: list of translations to score. Each translation
          should be tokenized into a list of tokens.
        lang(str) : The language generated code belongs to
    returns:
        A dicitonary with different metrics intact.
    """
    metrics_dict = {} #Update as in new metrics are added over here.
    metrics_dict["smoothed_bleu_4"] = compute_bleu(references,generated,smooth=True)
    metrics_dict["bleu_4"] = compute_bleu(references,generated,smooth=False)
    metrics_dict["parse_score"] = Parser(generated,lang)["parse_score"]
    return metrics_dict