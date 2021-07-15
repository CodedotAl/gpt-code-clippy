 
#Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import json
import argparse
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match


def python_process(tokens):
    new_tokens = []
    indent_count = 0
    num_tokens = len(tokens)
    tidx = 0
    while tidx < num_tokens:
        tok = tokens[tidx]
        tok = tok.strip()
        if tok in ["NEW_LINE"]:
            new_tokens.append("\n")
            if tidx + 1 < num_tokens:
                next_token = tokens[tidx + 1]
                if next_token == "INDENT":
                    indent_count += 1
                    tidx += 1
                elif next_token == "DEDENT":
                    indent_count -= 1
                    tidx += 1
            for ic in range(indent_count):
                new_tokens.append("\t")
        else:
            new_tokens.append(tok)
        tidx += 1
    return new_tokens
    pass


def php_process(tokens):
    new_tokens = []
    num_tokens = len(tokens)
    tidx = 0
    while tidx < num_tokens:
        tok = tokens[tidx]
        tok = tok.strip()
        if tok == "$":
            if tidx + 1 < num_tokens:
                tok += tokens[tidx + 1].strip()
                tidx += 1
                pass
            pass
        tidx += 1
        new_tokens.append(tok)
    return new_tokens


def language_specific_processing(tokens, lang):
    if lang == 'python':
        return python_process(tokens)
    elif lang == 'php':
        return php_process(tokens)
    else:
        return tokens


parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, required=True,
                    help='reference file')
parser.add_argument('--hyp', type=str, required=True,
                    help='hypothesis file')
parser.add_argument('--lang', type=str, required=True,
                    choices=['java', 'js', 'c_sharp', 'php', 'go', 'python', 'ruby'],
                    help='programming language')
parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                    help='alpha, beta and gamma')

args = parser.parse_args()

lang = args.lang
if lang == 'js':
    lang = 'javascript'
alpha, beta, gamma, theta = [float(x) for x in args.params.split(',')]

# preprocess inputs
references = [json.loads(x.strip())[lang] for x in open(args.ref, 'r', encoding='utf-8').readlines()]
hypothesis = [x.strip() for x in open(args.hyp, 'r', encoding='utf-8').readlines()]

assert len(hypothesis) == len(references)

# calculate ngram match (BLEU)
tokenized_hyps = [language_specific_processing(x.split(), lang) for x in hypothesis]
tokenized_refs = [[language_specific_processing(x.split(), lang) for x in reference] for reference in references]

ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

# calculate weighted ngram match
keywords = [x.strip() for x in open('keywords/' + lang + '.txt', 'r', encoding='utf-8').readlines()]


def make_weights(reference_tokens, key_word_list):
    return {token: 1 if token in key_word_list else 0.2 \
            for token in reference_tokens}


tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                for reference_tokens in reference] for reference in tokenized_refs]

weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

# calculate syntax match
syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

# calculate dataflow match
dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

# print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'. \
#      format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))
print('Ngram match:\t%.2f\nWeighted ngram:\t%.2f\nSyntax match:\t%.2f\nDataflow match:\t%.2f' % ( \
    ngram_match_score * 100, weighted_ngram_match_score * 100, syntax_match_score * 100, dataflow_match_score * 100))

code_bleu_score = alpha * ngram_match_score \
                  + beta * weighted_ngram_match_score \
                  + gamma * syntax_match_score \
                  + theta * dataflow_match_score

print('CodeBLEU score: %.2f' % (code_bleu_score * 100.0))