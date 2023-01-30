#!/usr/bin/env python3

from transformers import EvalPrediction
import torch
import numpy as np
import re
from collections import OrderedDict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def truncate_logits(tensors, minlen):
    # input: list
    # output: torch.Tensor
    shape = (len(tensors), len(tensors[0]))
    output = torch.zeros((*shape, minlen), device=DEVICE)
    for i, tensor in enumerate(tensors):
        for j, logits in enumerate(tensor):
            output[i, j] = torch.tensor(logits[:minlen], requires_grad=False)
    return output


def postprocess_qa(
        dataset,
        predictions,
        n_best_size=20,
        max_answer_length=20,
        top_k=5
    ):
    
    meaningful_pattern = re.compile('(\w|\d)+')
    # bos_residuals = re.compile('(?<=^)\W+')
    # eos_residuals = re.compile('\W+(?=$)')
    residuals = re.compile('(?<=^)\W+|\W+(?=$)') # bos_residuals | eos_residuals

    all_start_logits, all_end_logits = predictions[:2] # some models like e.g. BART return additional tensors
    all_predictions = OrderedDict()

    for start_logits, end_logits, entry in zip(all_start_logits, all_end_logits, dataset):

        # we get either np, or pt tensors
        if isinstance(start_logits, torch.Tensor): start_logits = start_logits.detach().numpy()
        if isinstance(end_logits, torch.Tensor): end_logits = end_logits.detach().numpy()

        prelim_predictions = []

        offset = entry['offset_mapping']
        offset_starts, offset_ends = [pair[0] for pair in offset], [pair[1] for pair in offset]

        # go through n_best_size greater logits (from -1 to -n_best_size - 1 with step -1)
        start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context
                if (
                    # start_index >= len(offset_mapping)
                    # or end_index >= len(offset_mapping)
                    offset[start_index] is None
                    or offset[end_index] is None
                ):
                    continue
                # don't consider answers with a length that is either < 0 or > max_answer_length
                if (end_index < start_index) or (end_index - start_index + 1 > max_answer_length):
                    continue
                prelim_predictions.append(
                    {
                        'offsets': (offset_starts[start_index], offset_ends[end_index]),
                        'score': start_logits[start_index] + end_logits[end_index],
                        'start_logit': start_logits[start_index],
                        'end_logit': end_logits[end_index],
                    }
                )

        # only keep n_best_size best predictions
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x['score'], reverse=True)[:n_best_size]

        predictions = []

        extracted = [] # remove duplicates
        context = entry['context']
        for pred in prelim_predictions:
            offsets = pred.pop('offsets')
            text = context[offsets[0]: offsets[1]]
            # heuristic for cleaning the empty preds
            if len(meaningful_pattern.findall(text)):
                # heuristic for cleaning the non-empty preds
                text = residuals.sub('', text)
                if not text in extracted:
                    pred['text'] = text
                    predictions.append(pred)
                    extracted.append(text)

        # append 'smoothing' pred in case we have no non-null ones
        if not len(predictions):
            predictions.append({'text': '', 'start_logit': 0.0, 'end_logit': 0.0, 'score': 0.0})

        if isinstance(top_k, int): best_predictions = predictions[:top_k]
        else: best_predictions = predictions[:1]
        all_predictions[entry['id']] = [
            prediction['text']
            for prediction in best_predictions
        ]

    formatted_predictions = [{'id': k, 'prediction_texts': v} for k, v in all_predictions.items()]

    references = [{'id': entry['id'], 'answers': entry['answers']} for entry in dataset]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def postprocess_classify(dataset, predictions, **_):

    # we get either np, or pt tensors
    if isinstance(predictions, torch.Tensor): predictions = predictions.detach().numpy()
    pred_labels = np.argmax(predictions, axis=1)

    references = dataset['label']
    return EvalPrediction(predictions=pred_labels, label_ids=references)


# TODO: IF "PHRASE", RETURN THE FIRST <= LENGTH N
# TODO: IF "PASSAGE", RETURN THE FIRST >= LENGTH N
# TODO: IF "MULTI", PREDICT RELEVANCE WITH A PRETRAINED LM AND RETURN THE RELEVANT ONES

# def postprocess_top_k(top_k, label):

#     match label:

#         case 'phrase': # the shortest
#             pass

#         case 'passage':
#             pass

#         case 'multi': # several
#             pass