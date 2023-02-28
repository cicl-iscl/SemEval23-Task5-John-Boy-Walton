#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import EvalPrediction
import torch
import numpy as np
import re
from collections import OrderedDict
from random import randint, sample

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#region Functional

def truncate_logits(tensors, minlen):
    # input: list
    # output: torch.Tensor
    shape = (len(tensors), len(tensors[0]))
    output = torch.zeros((*shape, minlen), device=DEVICE)
    for i, tensor in enumerate(tensors):
        for j, logits in enumerate(tensor):
            output[i, j] = torch.tensor(logits[:minlen], requires_grad=False)
    return output

    #region TextClassification

def postprocess_classify(dataset, predictions, **_):

    # we get either np, or pt tensors
    if isinstance(predictions, torch.Tensor): predictions = predictions.detach().numpy()
    pred_labels = np.argmax(predictions, axis=1)

    references = dataset['label']
    return EvalPrediction(predictions=pred_labels, label_ids=references)

    #endregion

    #region QA

def postprocess_qa(
        dataset,
        predictions,
        labels,
        n_best_size=20,
        min_answer_length=1,
        max_answer_length=20,
        top_k=5,
        remove_overlapping=True
    ):
    
    meaningful_pattern = re.compile('(\w|\d)+')
    bos_residuals = '(?<=^)[^([\'"\w]+'     # delete everything, except paired puncts
    eos_residuals = '[^.!?)\]\'"\w]+(?=$)'  # delete everything, except sentence break markers and paired puncts
    residuals = re.compile(f'{bos_residuals}|{eos_residuals}')

    def filter(text):
        text = text.strip()
        if not len(text): return
        if text.count('\'') % 2 != 0 : return              # '' never opened / closed
        if text.count('"') % 2 != 0: return                 # "" never opened / closed
        if text.count('(') != text.count(')'): return       # () never opened / closed
        if text.count('[') != text.count(']'): return       # [] never opened / closed
        if len(meaningful_pattern.findall(text)):
            text = residuals.sub('', text)
            return text

    all_start_logits, all_end_logits = predictions[:2] # some models like e.g. BART return additional tensors
    all_predictions = OrderedDict()

    for start_logits, end_logits, label, entry in zip(all_start_logits, all_end_logits, labels, dataset):

        # we get either np, or pt tensors
        if isinstance(start_logits, torch.Tensor): start_logits = start_logits.detach().numpy()
        if isinstance(end_logits, torch.Tensor): end_logits = end_logits.detach().numpy()

        prelim_predictions = []

        offset = entry['offset_mapping']
        offset_starts, offset_ends = [pair[0] for pair in offset], [pair[1] for pair in offset]

        # # go through n_best_size greater logits (from -1 to -n_best_size - 1 with step -1)
        start_indices = np.argsort(start_logits) #[-1: -n_best_size - 1: -1].tolist()
        end_indices = np.argsort(end_logits) #[-1: -n_best_size - 1: -1].tolist()
        for start_index in start_indices:
            for end_index in end_indices:
                # don't consider out-of-scope answers, because the indices either are out of bounds or correspond
                # to part of the input_ids that are not in the context
                if (
                    offset[start_index] is None
                    or offset[end_index] is None
                ): continue
                # # don't consider answers where end offset is 0
                # if offset_ends[end_index] == 0: continue
                # # don't consider answers with a length that is either <= 0 or > `max_answer_length` or < `min_answer_length`
                # if (
                #     (end_index <= start_index )
                #     or (end_index - start_index + 1 > max_answer_length)
                #     or (end_index - start_index + 1 < min_answer_length)
                # ): continue
                prelim_predictions.append(
                    {
                        'offsets': (offset_starts[start_index], offset_ends[end_index]),
                        'score': start_logits[start_index] + end_logits[end_index],
                        'start_logit': start_logits[start_index],
                        'end_logit': end_logits[end_index],
                    }
                )

        # only keep n_best_size best predictions
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)

        # now quite often the best predictions are very similar except for a couple of extra words;
        # so to exclude overlapping and have more options we keep only unique ranges that weren't (partially) extracted yet
        # NB! in case of overlapping: 
        #   -- we'll keep the shortest chunk if label is "phrase" or "multi"
        #   -- we'll keep the longest chunk if label is "passage" (make it greedy)

        context = entry['context']
        def process_prediction(prediction):
            offsets = prediction['offsets']
            text = context[offsets[0]: offsets[1]]
            # heuristic for cleaning non-empty preds
            text_ = filter(text)
            if not text_ is None:
                simple_tokenized = text_.split()
                if (
                    len(simple_tokenized) >= min_answer_length
                    and len(simple_tokenized) <= max_answer_length
                ):
                    prediction['text'] = text_
                    return prediction


        allocated = set()

        if remove_overlapping:
        
            predictions = []
            for prelim_prediction in prelim_predictions:
                prelim_prediction = process_prediction(prelim_prediction)
                if prelim_prediction is None: continue
                offsets = prelim_prediction['offsets']
                prelim_range = set(range(*offsets))
                overlapping_offsets = allocated & prelim_range
                # no overlapping => can add if passes the filters
                if not len(overlapping_offsets):
                    predictions.append(prelim_prediction)
                    allocated.update(prelim_range)
                # overlapping => keep the shortest / the longest chunk
                else:
                    for pred in predictions:
                        offsets = pred['offsets']
                        pred_range = set(range(*offsets))
                        overlapping_offsets = prelim_range & pred_range
                        if len(overlapping_offsets):
                            # keep the longest chunk
                            if (label == 'passage') and (len(prelim_range) > len(pred_range)):
                                # add new indices
                                allocated.update(prelim_range)
                                predictions.remove(pred)
                                predictions.append(prelim_prediction)
                                break
                            # keep the shortest chunk (upd: doesn't work)
                            # elif (label != 'passage') and (len(prelim_range) < len(pred_range)):
                            #     # remove excessive indices
                            #     difference = pred_range - prelim_range
                            #     allocated -= difference
                            #     predictions.remove(pred)
                            #     predictions.append(prelim_prediction)
                            #     counter += 1; break
                if len(predictions) == n_best_size: break

        else:

            predictions = []
            for prelim_prediction in prelim_predictions:
                prelim_prediction = process_prediction(prelim_prediction)
                if prelim_prediction is None: continue
                predictions.append(prelim_prediction)
                if len(predictions) == n_best_size: break



        # append 'smoothing' pred in case we have no non-null ones
        if not len(predictions): predictions.append(
                {'offsets': (-1, -1), 'score': 0, 'start_logit': 0, 'end_logit': 0, 'text': ''}
                )

        if isinstance(top_k, int): best_predictions = predictions[:top_k]
        else: best_predictions = predictions[:1]
        all_predictions[entry['id']] = [
            prediction['text']
            for prediction in best_predictions
        ]

    formatted_predictions = [{'id': k, 'prediction_texts': v} for k, v in all_predictions.items()]

    references = [{'id': entry['id'], 'answers': entry['answers']} for entry in dataset]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def postprocess_top_k(top_k, label):

    # match label:
    #     case 'phrase': pass # the shortest
    #     case 'passage': pass # the longest
    #     case 'multi': pass # concatenate

    if label != 'multi': return top_k[0] # it should already have respective length after `postprocess_qa()`
    else: # concatenate predictions otherwise
        # as we don't know how many predictions are really relevant,
        # we'll pick them at random
        k = len(top_k)
        if len(top_k) < 2: return top_k[0] # == 1
        rand_n = randint(2, k)
        # we need to keep scored order, so we can't use `sample()` directly
        indices = range(k)
        rand_indices = sorted(sample(indices, rand_n))
        rand_preds = [pred for i, pred in enumerate(top_k) if i in rand_indices]
        return ', '.join(rand_preds)
    
    #endregion

#endregion