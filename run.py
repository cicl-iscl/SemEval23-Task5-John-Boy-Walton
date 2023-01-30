#!/usr/bin/env python3

import torch
import gc
import os
import argparse
import json

from utils.postprocess import *
from utils.dataset import compose_datasets
from web_trainer import *
from downstream.answer import build_model as build_qa, answer
from downstream.summarize import build_model as build_summarizer, summarize
from downstream.classify import build_model as build_classifier, classify

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.cuda.set_per_process_memory_fraction(0.5, 0) # so that torch doesn't allocate the whole RAM
    torch.cuda.empty_cache()
    gc.collect()
else: DEVICE = 'cpu'


def read_instructions(instructions_dir):
    instructions = []
    for root, _, filenames in os.walk(instructions_dir):
        for filename in filenames:
            if not 'template' in filename:
                path = os.path.join(root, filename)
                with open(path) as i: instruction = json.load(i)
                instructions.append(instruction)
    return instructions


def run(
        qa_instructions_dir, 
        classification_instructions_dir,
        output_dir,
        X_train=None,
        X_dev=None,
        X_test=None,
        mode='test', # 'train' / 'test' 
        use_summarization=True,
        summarize_only_on_cuda=True,
        batch_size=8,
        save_datasets=False,
        saved_datasets_dir='webis22_prepared',
        **postprocess_qargs
    ):

    if mode == 'train' and X_train is None: raise AttributeError('Can\'t train without X_train.')
    if mode == 'test' and X_test is None: raise AttributeError('Can\'t test without X_test.')

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    #region QA

    '''
    ...
    '''

    # leave only 'question', 'context' for qa to accept it
    if X_test is not None:
        to_remove = X_test.column_names
        to_remove.remove('question'); to_remove.remove('context')
        X_test_ = X_test.remove_columns(to_remove)
    else: X_test_ = None
    # QA models don't like that the label is a list
    X_train_ = X_train.remove_columns('label') if X_train is not None else None
    X_dev_ = X_dev.remove_columns('label') if X_dev is not None else None
    
    qa_start_logits, qa_end_logits = [], []
    qa_instructions = read_instructions(qa_instructions_dir)
    for instruction in qa_instructions:
        if not instruction['use']: continue
        print(instruction['name'])
        qa = build_qa(instruction, X_train_, X_dev_, mode)
        if X_test_ is not None:
            X_test_ = X_test_.map(
                answer, batched=True, batch_size=batch_size,
                fn_kwargs={'qa': qa}
            )
            qa_start_logits.append(X_test_['start_logits'])
            qa_end_logits.append(X_test_['end_logits'])

    if X_test_ is not None:

        # now we have our logits in lists because we retrieve them from a datasets.Dataset instance
        minlen = min([len(logits) for start_logits in qa_start_logits for logits in start_logits])
        qa_start_logits = truncate_logits(qa_start_logits, minlen)
        qa_end_logits = truncate_logits(qa_end_logits, minlen)

        start_logits_ensemble = torch.mean(qa_start_logits, axis=0).cpu()
        end_logits_ensemble = torch.mean(qa_end_logits, axis=0).cpu()

        output = postprocess_qa(X_test, (start_logits_ensemble, end_logits_ensemble), **postprocess_qargs)
        pred_spoilers = output.predictions

        top_k_path = os.path.join(output_dir, 'top_k.json')
        with open(top_k_path, 'w') as k: json.dump(pred_spoilers, k, indent=2)

    #endregion

    #region Summarization

    '''
    ...
    '''

    summarization_allowed = (not summarize_only_on_cuda) or (summarize_only_on_cuda and 'cuda' in DEVICE)
    if use_summarization and summarization_allowed:

        model, tokenizer = build_summarizer(DEFAULT_S2S_MODEL_NAME)

        if X_train is not None and not 'summarized' in X_train.column_names:
            X_train = X_train.map(
                summarize, input_columns='context', batched=True, batch_size=batch_size, 
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if X_dev is not None and not 'summarized' in X_dev.column_names:
            X_dev = X_dev.map(
                summarize, input_columns='context', batched=True, batch_size=batch_size,
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if X_test is not None and not 'summarized' in X_test.column_names:
            X_test = X_test.map(
                summarize, input_columns='context', batched=True, batch_size=batch_size,
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if save_datasets:

            if not os.path.exists(saved_datasets_dir):
                os.mkdir(saved_datasets_dir)

            if X_train is not None: X_train.to_json(os.path.join(saved_datasets_dir, 'train.jsonl'))
            if X_dev is not None: X_dev.to_json(os.path.join(saved_datasets_dir, 'dev.jsonl'))
            if X_test is not None: X_test.to_json(os.path.join(saved_datasets_dir, 'test.jsonl'))

    #endregion

    #region Classification

    '''
    ...
    '''

    # leave only 'label', 'context' / 'summarized' --> 'text' for classifier to accept it
    if X_test is not None:
        text_column = 'summarized' if 'summarized' in X_test.column_names else 'title' # 'context'
        to_remove = X_test.column_names
        to_remove.remove(text_column)
        X_test_ = X_test.remove_columns(to_remove).rename_column(text_column, 'text')
    else: X_test_ = None

    cl_logits = []
    classification_instructions = read_instructions(classification_instructions_dir)
    for instruction in classification_instructions:
        if not instruction['use']: continue
        print(instruction['name'])
        label2id = lambda labels: {'label': [instruction['label_mapping'][label] for label in labels]}
        mapper = lambda data: data.map(label2id, batched=True, input_columns='label') if not data is None else None
        classifier = build_classifier(instruction, mapper(X_train), mapper(X_dev), mode)
        if X_test_ is not None:
            X_test_ = X_test_.map(
                classify, batched=True, batch_size=batch_size,
                fn_kwargs={'classifier': classifier}
            )
            cl_logits.append(X_test_['logits'])

    if X_test is not None:

        cl_logits = torch.tensor(cl_logits, requires_grad=False)
        logits_ensemble = torch.mean(cl_logits, axis=0)

        output = postprocess_classify(X_test, logits_ensemble)
        pred_label_ids = output.predictions

        id2label = lambda label_id: classifier.model.config.id2label[label_id] # use latest
        pred_labels = [
            {'id': id, 'label': id2label(label_id)}
            for id, label_id in zip(X_test['id'], pred_label_ids)
        ]
        
        label_path = os.path.join(output_dir, 'labels.json')
        with open(label_path, 'w', encoding='utf-8-sig') as l: json.dump(pred_labels, l, indent=2)

    #endregion

    if X_test is not None:

        answers = []

        pred_spoilers_ = {entry['id']: entry['prediction_texts'] for entry in pred_spoilers}
        pred_labels_ = {entry['id']: entry['label'] for entry in pred_labels}

        def find_by_id(preds, id):
            id_idx = list(preds.keys()).index(id)
            return list(preds.values())[id_idx]

        for id in X_test['id']:

            top_k = find_by_id(pred_spoilers_, id)
            label = find_by_id(pred_labels_, id)
            # final_spoiler = postprocess_top_k(top_k, label)
            final_spoiler = top_k[0]

            answer = {
                'uuid': id,
                'spoilerType': label,
                'spoiler': final_spoiler
            }
            answers.append(answer)

        answer_path = os.path.join(output_dir, 'run.jsonl')
        with open(answer_path, 'w', encoding='utf-8-sig') as ans: 
            for answer in answers:
                json.dump(answer, ans)
                ans.write('\n')


def main():

    parser = argparse.ArgumentParser(
        prog='WebEnsemble',
        description='Ensemble model for SemEval23 Task 5'
    )

    parser.add_argument('input_dir', default='webis22_run')
    parser.add_argument('output_dir', default='out')
    parser.add_argument('-i', '--instructions_dir', required=False, default='instructions')
    parser.add_argument('-p', '--preprocess_mode', required=False, default='0', choices=['0', '1', '2']) #, help =
    parser.add_argument('-m', '--mode', required=False, default='test', choices=['train', 'test'])
    parser.add_argument('-s', '--summarize', required=False, default='True')
    parser.add_argument('-oc', '--summarize_only_on_cuda', required=False, default='True')
    parser.add_argument('-bs', '--batch_size', required=False, default='8')
    parser.add_argument('-save', '--save_datasets', required=False, default='False')
    parser.add_argument('-save_dir', '--saved_datasets_dir', required=False, default='webis22_summarized')

    args = parser.parse_args()

    match args.preprocess_mode:
        case '0': preprocess_train = True; preprocess_test = True # for initial training (and prediction) // webis22_original
        case '1': preprocess_train = False; preprocess_test = True # for prediction after training        // webis22_run
        case '2': preprocess_train = False; preprocess_test = False # for evaluation and tests            // webis22_summarized

    X_train, X_dev, X_test = compose_datasets(args.input_dir, preprocess_train, preprocess_test)

    qa_dir = os.path.join(args.instructions_dir, 'QA') # 'instructions/QA'
    if not os.path.exists(qa_dir): 
        raise ValueError('There should be a \'QA\' directory in the instructions directory.')
    classify_dir = os.path.join(args.instructions_dir, 'TextClassification') # 'instructions/TextClassification'
    if not os.path.exists(classify_dir): 
        raise ValueError('There should be a \'TextClassification\' directory in the instructions directory.')

    postprocess_qargs_path = os.path.join(args.instructions_dir, 'postprocess_qargs.json')
    if os.path.exists(postprocess_qargs_path):
        with open(postprocess_qargs_path) as f: postprocess_qargs = json.load(f)
    else: postprocess_qargs = {}

    run(
        qa_instructions_dir=qa_dir,
        classification_instructions_dir=classify_dir,
        output_dir=args.output_dir,
        X_train=X_train,
        X_dev=X_dev,
        X_test=X_test,
        mode=args.mode,
        use_summarization=args.summarize == 'True',
        summarize_only_on_cuda=args.summarize_only_on_cuda == 'True',
        batch_size=int(args.batch_size),
        save_datasets=args.save_datasets == 'True',
        saved_datasets_dir=args.saved_datasets_dir,
        **postprocess_qargs
    )

# EXAMPLE USAGE: python3 run.py webis22_run out -i instructions -p 1 -s True -save True -save_dir webis22_summarized
if __name__ == '__main__':
    main()

# TODO: EVALUATION

# TODO: ADD LINKS TO THE MODELS IN JSONS