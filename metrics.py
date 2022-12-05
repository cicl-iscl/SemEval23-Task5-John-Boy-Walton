import numpy as np
from nltk.translate.bleu_score import sentence_bleu


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