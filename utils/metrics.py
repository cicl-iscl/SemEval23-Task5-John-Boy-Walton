#!/usr/bin/env python3

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score


def bleu(eval_preds):
    all_answers = [ref['answers'] for ref in eval_preds.label_ids]
    all_references = [' '.join(
            [answer['text'] for answer in answers]
        ) for answers in all_answers
    ]
    all_candidates = [ref['prediction_texts'] for ref in eval_preds.predictions]
    BLEU = np.mean([
        sentence_bleu(references, candidates)
        for references, candidates in zip(all_references, all_candidates)
    ])
    return {'eval_bleu': BLEU}


def accuracy(eval_preds):
    true = eval_preds.label_ids
    pred = eval_preds.predictions
    acc = accuracy_score(true, pred)
    return {'eval_accuracy': acc}