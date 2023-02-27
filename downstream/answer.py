#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import os

from web_trainer import *

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#region Functional

def fine_tune(instruction, X_train=None, X_dev=None):
    model = AutoModelForQuestionAnswering.from_pretrained(instruction['input_model_path']).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(instruction['input_model_path'])
    training_args = WebQATrainingArguments(**instruction['training_kwargs']).configured
    trainer = WebQATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_dev,
        **instruction['trainer_kwargs']
    ).configured
    trainer.train()
    dir_name = os.path.dirname(instruction['output_model_path'])
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    trainer.save_model(dir_name)


def build_model(instruction, X_train=None, X_dev=None, mode='train'):
    if instruction['fine-tune'] and mode == 'train': fine_tune(instruction, X_train, X_dev)
    model = AutoModelForQuestionAnswering.from_pretrained(instruction['output_model_path']).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(instruction['output_model_path'])
    qa = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return qa


def retrieve_answer(X_test, qa):
    inputs = qa.tokenizer(
        # will concatenate question and context
        list(X_test['question']),
        list(X_test['context']),
        truncation='only_second', # truncate only context
        padding='longest', # True
        # return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad(): output = qa.model(**inputs)
    return {
        'start_logits': output['start_logits'],
        'end_logits': output['end_logits']
    } # datasets.Dataset.map() requires the function to return a dict

#endregion