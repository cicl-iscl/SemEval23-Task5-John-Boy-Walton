#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 0
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

import torch
import gc
import os
import argparse
import json

from utils.postprocess import *
from utils.dataset import compose_datasets
from web_trainer import *
from downstream.answer import build_model as build_qa, retrieve_answer
from downstream.summarize import build_model as build_summarizer, summarize
from downstream.classify import build_model as build_classifier, classify

#endregion


#region Cleanup

def clean_cuda():
    torch.cuda.empty_cache()
    gc.collect()

if torch.cuda.is_available():
    DEVICE = 'cuda'
    # torch.cuda.set_per_process_memory_fraction(0.75, 0) # so that torch doesn't allocate the whole RAM
    clean_cuda()
else: DEVICE = 'cpu'


def cleanup(dir):
    for root, dirs, files in os.walk(dir, topdown=False):
        for file_name in files:
            file_name_ = os.path.join(root, file_name)
            if os.path.exists(file_name_):
                os.remove(file_name_)
        for dir_name in dirs:
            dir_name_ = os.path.join(root, dir_name)
            if os.path.exists(dir_name_):
                os.rmdir(dir_name_)
    if os.path.exists(dir): os.rmdir(dir)

#endregion


def read_instructions(instructions_dir):
    instructions = []
    for root, _, filenames in os.walk(instructions_dir):
        for filename in filenames:
            if not 'template' in filename:
                path = os.path.join(root, filename)
                with open(path) as i: instruction = json.load(i)
                instructions.append(instruction)
    return instructions


#region Run

def run(
        qa_instructions_dir, 
        classification_instructions_dir,
        output_dir,
        subtask,
        X_train=None,
        X_dev=None,
        X_test=None,
        mode='test', # 'train' / 'test' 
        use_summarization=False,
        summarize_only_on_cuda=True,
        save_datasets=False,
        saved_datasets_dir='webis22_prepared',
        **postprocess_qargs
    ):

    '''
    Steps:
        1. Summarize (optional);
        2. Predict label either with summarized text or with title;
        3. Retrieve spoiler;
        4. Postprocess spoiler with predicted label.
    '''

    if mode == 'train' and X_train is None: raise AttributeError('Can\'t train without X_train.')
    if mode == 'test' and X_test is None: raise AttributeError('Can\'t test without X_test.')

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    #region Summarization

    '''
    For classification we will use either summarized texts or titles.
    As classification goes first, we should prepared summarized texts if they'll be used.
    '''

    summarization_allowed = (not summarize_only_on_cuda) or (summarize_only_on_cuda and 'cuda' in DEVICE)
    if use_summarization and summarization_allowed:

        BATCH_SIZE = 8

        model, tokenizer = build_summarizer(DEFAULT_S2S_MODEL_NAME)

        if X_train is not None and not 'summarized' in X_train.column_names:
            X_train = X_train.map(
                summarize, input_columns='context', batched=True, batch_size=BATCH_SIZE, 
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if X_dev is not None and not 'summarized' in X_dev.column_names:
            X_dev = X_dev.map(
                summarize, input_columns='context', batched=True, batch_size=BATCH_SIZE,
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if X_test is not None and not 'summarized' in X_test.column_names:
            X_test = X_test.map(
                summarize, input_columns='context', batched=True, batch_size=BATCH_SIZE,
                fn_kwargs={'summarizer': model, 'tokenizer': tokenizer}
            )

        if save_datasets:

            if not os.path.exists(saved_datasets_dir):
                os.mkdir(saved_datasets_dir)

            if X_train is not None: X_train.to_json(os.path.join(saved_datasets_dir, 'train.jsonl'))
            if X_dev is not None: X_dev.to_json(os.path.join(saved_datasets_dir, 'validation.jsonl'))
            if X_test is not None: X_test.to_json(os.path.join(saved_datasets_dir, 'input.jsonl'))

    if DEVICE == 'cuda': clean_cuda()

    #endregion

    #region Classification

    '''
    Use ensemble approach to classification; gather logits from each model, calculate mean, then postprocess.
    '''

    # leave only 'context' / 'summarized' --> 'text' for classifier to accept it
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
        X_train_ = mapper(X_train)
        # 'context' / 'summarized' --> 'text'
        if X_train_ is not None:
            text_column = 'summarized' if 'summarized' in X_train_.column_names else 'title' # 'context'
            X_train_ = X_train_.rename_column(text_column, 'text')
        X_dev_ = mapper(X_dev)
        if X_dev_ is not None:
            text_column = 'summarized' if 'summarized' in X_dev_.column_names else 'title' # 'context'
            X_dev_ = X_dev_.rename_column(text_column, 'text')
        classifier = build_classifier(instruction, X_train_, X_dev_, mode)

        # delete checkpoints and logs to clean up the space
        if instruction['training_kwargs'].get('output_dir') is not None:
            cleanup(instruction['training_kwargs']['output_dir'])
        if instruction['training_kwargs'].get('logging_dir') is not None:
            cleanup(instruction['training_kwargs']['logging_dir'])

        # if X_test_ is not None:
        if X_test_ is not None:
            X_test_ = X_test_.map(
                classify, batched=True, batch_size=instruction['test_batch_size'],
                fn_kwargs={'classifier': classifier}
            )
            cl_logits.append(X_test_['logits'])

        if DEVICE == 'cuda': clean_cuda()

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
        
        if mode == 'train':
            label_path = os.path.join(output_dir, 'labels.json')
            with open(label_path, 'w', encoding='utf8') as l: json.dump(pred_labels, l, indent=4)

    #endregion

    #region QA

    '''
    Use ensemble approach to QA; gather logits from each model, truncate / pad, calculate mean, then postprocess.
    '''

    if subtask == 2:

        # leave only 'question', 'context' for qa to accept it
        if X_test is not None:
            to_remove = X_test.column_names
            to_remove.remove('question'); to_remove.remove('context')
            X_test_ = X_test.remove_columns(to_remove)
        else: X_test_ = None
        # QA models don't like that the label is a string
        X_train_ = X_train.remove_columns('label') if X_train is not None else None
        X_dev_ = X_dev.remove_columns('label') if X_dev is not None else None
        
        qa_start_logits, qa_end_logits = [], []
        qa_instructions = read_instructions(qa_instructions_dir)
        for instruction in qa_instructions:

            if not instruction['use']: continue
            print(instruction['name'])
            qa = build_qa(instruction, X_train_, X_dev_, mode)

            # delete checkpoints and logs to clean up the space
            if instruction['training_kwargs'].get('output_dir') is not None:
                cleanup(instruction['training_kwargs']['output_dir'])
            if instruction['training_kwargs'].get('logging_dir') is not None:
                cleanup(instruction['training_kwargs']['logging_dir'])

            if X_test_ is not None:
                X_test_ = X_test_.map(
                    retrieve_answer, batched=True, batch_size=instruction['test_batch_size'],
                    fn_kwargs={'qa': qa}
                )
                qa_start_logits.append(X_test_['start_logits'])
                qa_end_logits.append(X_test_['end_logits'])

        if DEVICE == 'cuda': clean_cuda()

        if X_test_ is not None:

            # now we have our logits in lists because we retrieve them from a datasets.Dataset instance
            minlen = min([len(logits) for start_logits in qa_start_logits for logits in start_logits])
            qa_start_logits = truncate_logits(qa_start_logits, minlen)
            qa_end_logits = truncate_logits(qa_end_logits, minlen)

            start_logits_ensemble = torch.mean(qa_start_logits, axis=0).cpu()
            end_logits_ensemble = torch.mean(qa_end_logits, axis=0).cpu()

            labels = [pred_label['label'] for pred_label in pred_labels]
            output = postprocess_qa(X_test, (start_logits_ensemble, end_logits_ensemble), labels, **postprocess_qargs)
            pred_spoilers = output.predictions

            if mode == 'train':
                top_k_path = os.path.join(output_dir, 'top_k.json')
                with open(top_k_path, 'w', encoding='utf8') as k: json.dump(pred_spoilers, k, indent=4)

    #endregion

    #region Finalize

    '''
    Retrieve the best spoiler from top-k with predicted labels.
    '''

    if X_test is not None:

        answers = []

        pred_labels_ = {entry['id']: entry['label'] for entry in pred_labels}
        if subtask == 2: pred_spoilers_ = {entry['id']: entry['prediction_texts'] for entry in pred_spoilers}

        def find_by_id(preds, id):
            id_idx = list(preds.keys()).index(id)
            return list(preds.values())[id_idx]

        for id in X_test['id']:

            label = find_by_id(pred_labels_, id)
            if subtask == 2:
                top_k = find_by_id(pred_spoilers_, id)
                final_spoiler = postprocess_top_k(top_k, label)
                # final_spoiler = top_k[0]

            answer = {
                'uuid': id,
                'spoilerType': label
            } if subtask == 1 else {
                'uuid': id,
                'spoilerType': label,
                'spoiler': final_spoiler
            }
            answers.append(answer)

        answer_path = os.path.join(output_dir, 'run.jsonl')
        with open(answer_path, 'w', encoding='utf8') as ans: 
            for answer in answers:
                json.dump(answer, ans)
                ans.write('\n')

    #endregion

#endregion


#region Executive

def main():

    parser = argparse.ArgumentParser(
        prog='WebEnsemble',
        description='Ensemble model for SemEval23 Task 5'
    )

    parser.add_argument('input_dir', default='./webis22_run')
    parser.add_argument('output_dir', default='./out')
    parser.add_argument('subtask', default='2', choices=['1', '2'])
    # parser.add_argument('-i', '--instructions_dir', required=False, default='/WebSemble/instructions_docker')
    parser.add_argument('-i', '--instructions_dir', required=False, default='./instructions_local')
    parser.add_argument('-p', '--preprocess_mode', required=False, default='1', choices=['0', '1', '2'])
    parser.add_argument('-m', '--mode', required=False, default='test', choices=['train', 'test'])
    parser.add_argument('-s', '--summarize', required=False, default='False', choices=['True', 'False'])
    parser.add_argument('-oc', '--summarize_only_on_cuda', required=False, default='True', choices=['True', 'False'])
    parser.add_argument('-save', '--save_datasets', required=False, default='False', choices=['True', 'False'])
    parser.add_argument('-save_dir', '--saved_datasets_dir', required=False, default='./webis22_summarized')

    args = parser.parse_args()

    # match args.preprocess_mode:
    #     case '0': preprocess_train = True; preprocess_test = True # for initial training (and prediction) // webis22_original
    #     case '1': preprocess_train = False; preprocess_test = True # for prediction after training        // webis22_run
    #     case '2': preprocess_train = False; preprocess_test = False # for evaluation and tests            // webis22_summarized

    if args.preprocess_mode == '0': preprocess_train = True; preprocess_test = True # for initial training (and prediction)   // webis22_original
    elif args.preprocess_mode == '1': preprocess_train = False; preprocess_test = True # for prediction after training        // webis22_run
    elif args.preprocess_mode == '2': preprocess_train = False; preprocess_test = False # for evaluation and tests            // webis22_summarized

    X_train, X_dev, X_test = compose_datasets(args.input_dir, preprocess_train, preprocess_test, mode=args.mode)

    qa_dir = os.path.join(args.instructions_dir, 'QA') # 'instructions_local/QA'
    if not os.path.exists(qa_dir): 
        raise ValueError('There should be a \'/QA\' directory in the instructions directory.')
    classify_dir = os.path.join(args.instructions_dir, 'TextClassification') # 'instructions_local/TextClassification'
    if not os.path.exists(classify_dir): 
        raise ValueError('There should be a \'/TextClassification\' directory in the instructions directory.')

    postprocess_qargs_path = os.path.join(args.instructions_dir, 'postprocess_qargs.json')
    if os.path.exists(postprocess_qargs_path):
        with open(postprocess_qargs_path) as f: postprocess_qargs = json.load(f)
    else: postprocess_qargs = {}

    '''
    % python3 run.py ./webis22_run ./out 2 -m train
    On Apple M1, CPU:

    deberta-v3-base-tasksource-nli
    100%|████████████████████████████████████████████████████████████████████████████████| 200/200 [00:26<00:00,  7.60ba/s]
    distilbert-base-uncased-webis22
    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:06<00:00, 14.50ba/s]
    bert-base-uncased-MNLI-webis22
    100%|████████████████████████████████████████████████████████████████████████████████| 200/200 [00:18<00:00, 10.89ba/s]
    roberta-base-squad2
    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [04:14<00:00,  2.55s/ba]
    bert-large-uncased-whole-word-masking-finetuned-squad
    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [13:53<00:00,  8.34s/ba]
    bart-base-webis22
    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [14:15<00:00,  8.56s/ba]
    distilbert-base-cased-distilled-squad
    100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [01:52<00:02,  1.13s/ba]
    '''

    run(
        qa_instructions_dir=qa_dir,
        classification_instructions_dir=classify_dir,
        output_dir=args.output_dir,
        subtask=int(args.subtask),
        X_train=X_train,
        X_dev=X_dev,
        X_test=X_test,
        mode=args.mode,
        use_summarization=args.summarize == 'True',
        summarize_only_on_cuda=args.summarize_only_on_cuda == 'True',
        save_datasets=args.save_datasets == 'True',
        saved_datasets_dir=args.saved_datasets_dir,
        **postprocess_qargs
    )


# EXAMPLE USAGE: % python3 run.py ./webis22_run ./out 2 -i instructions_local -p 1 -s True -save True -save_dir ./webis22_summarized
if __name__ == '__main__':
    main()

#endregion