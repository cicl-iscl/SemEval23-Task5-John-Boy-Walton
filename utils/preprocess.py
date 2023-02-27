#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import AutoTokenizer
import torch
import pandas as pd
import json

from web_trainer import DEFAULT_MODEL_NAME

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#region Functional

def read_json(path):
    return [json.loads(i) for i in open(path)]


# see https://www.tensorflow.org/datasets/catalog/squad
def squad_format(path, mode='train'): # 'train' / 'test'

    def retrieve_answer_starts(obs):
        prefix_len = len(obs['targetTitle']) + 2 # we concatenate with ' - ' in between but will count the second ' ' later
        return [
            prefix_len + 
            len(''.join(obs['targetParagraphs'][:paragraph_index])) + 
            spoiler_start +
            paragraph_index + 1 # because we concatenate title and paragraphs with ' ' in between
            for paragraph_index, spoiler_start
            in [spoiler_start_[0] for spoiler_start_ in obs['spoilerPositions']]
        ]

    def retrieve_answers(obs, mode=mode):
        if mode == 'train':
            return [{
                'answer_start': spoiler_start,
                'text': spoiler
            } for spoiler, spoiler_start in zip(obs['spoiler'], retrieve_answer_starts(obs))]
        elif mode == 'test':
            return 'not available'

    return pd.DataFrame([
        {
            'id': obs['uuid'], 
            'title': obs['targetTitle'], 
            'question': ' '.join(obs['postText']), 
            'context': obs['targetTitle'] + ' - ' + (' '.join(obs['targetParagraphs'])), 
            'answers': retrieve_answers(obs, mode=mode),
            # 'label' doesn't belong to SQuAD for classification, will be ignored in QA but used in classification
            'label': obs['tags'][0] if mode == 'train' else 'not available'
        } for obs in read_json(path)
        ])


def from_squad(
        squad_data, 
        tokenizer=AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME),
        mode='train', # 'train' / 'test'
        # batch_size=None
    ):

    questions = list(squad_data['question'])
    contexts = list(squad_data['context'])
    
    inputs = tokenizer(
        # will concatenate question and context
        questions,
        contexts,
        truncation='only_second',
        padding='max_length',
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(DEVICE)

    # points to start and end indices of the tokens in the raw input
    offset_mapping = inputs['offset_mapping']
    answers = list(squad_data['answers'])
    start_positions = []
    end_positions = []

    if mode == 'train':

        for i, offset in enumerate(offset_mapping):

            offset_starts, offset_ends = [pair[0] for pair in offset], [pair[1] for pair in offset]
            for answer in answers[i]:

                answer_start_positions = []
                answer_end_positions = []

                start_char = answer['answer_start']
                end_char = answer['answer_start'] + len(answer['text'])
                # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.sequence_ids
                    # None for special tokens added around or between sequences
                    # 0 for tokens corresponding to words in the first sequence ==> question
                    # 1 for tokens corresponding to words in the second sequence when a pair of sequences was jointly encoded ==> context
                # e.g.  question --> ['CSL', qt0, qt1, qt2, 'CLS']
                #       context --> ['CSL', ct0, ct1, ct2, ct3, ct4, 'CLS']
                #       sequence is concatenation --> ['CSL', qt0, qt1, qt2, 'CLS', 'CSL', ct0, ct1, ct2, ct3, ct4, 'CLS']
                #       apply padding --> ['CSL', qt0, qt1, qt2, 'CLS', 'CSL', ct0, ct1, ct2, ct3, ct4, 'CLS', 'PAD', 'PAD']
                #       sequence_ids() --> [None, 0,   0,   0,    None,  None, 1,   1,   1,   1,   1,    None,  None,  None]
                sequence_ids = inputs.sequence_ids(i)

                context_start = sequence_ids.index(1) # the first token with label 1 ==> in context
                context_end = len(sequence_ids) - 1 - sequence_ids.index(1) # the last (first in reverse) token with label 1 ==> in context

                # (0, 0) if the answer is not fully inside the context (e.g. in case of truncation)
                if not set(range(start_char, end_char)).issubset(set(range(offset_starts[context_start], offset_ends[context_end]))):
                    start_position = 0
                    end_position = 0
                else: # otherwise it's the start and end token positions
                    # narrow the scope
                    while offset_starts[context_start] <= start_char: context_start += 1
                    start_position = context_start - 1
                    while offset_ends[context_end] >= end_char: context_end -= 1
                    end_position = context_end + 1

                answer_start_positions.append(start_position)
                answer_end_positions.append(end_position)

            start_positions.append(answer_start_positions)
            end_positions.append(answer_end_positions)

        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions

    return inputs

#endregion