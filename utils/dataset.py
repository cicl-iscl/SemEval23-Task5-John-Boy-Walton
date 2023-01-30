#!/usr/bin/env python3

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

from datasets import Dataset

from utils.preprocess import squad_format, from_squad


def compose_datasets(dir, preprocess_train=True, preprocess_test=True):

    # we have two cases here: or we read and preprocess provided datasets, 
    # or we use those preprocessed in advance

    if preprocess_train:

        train_path = os.path.join(dir, 'train.jsonl')
        if os.path.exists(train_path):

            X_train = Dataset.from_pandas(
                squad_format(train_path)
            )
            # could leave only 'input_ids', 'attention_mask', 'start_positions', 'end_positions'
            # but will not remove columns as the unused ones are ignored
            # and the 'context', but will help us in postprocessing
            X_train = X_train.map(from_squad, batched=True)#, remove_columns=X.column_names)

        else: X_train = None
            
        dev_path = os.path.join(dir, 'validation.jsonl')
        if os.path.exists(train_path):

            X_dev = Dataset.from_pandas(
                squad_format(train_path)
            )
            X_dev = X_dev.map(from_squad, batched=True)

        else: X_dev = None, None

    else:

        train_path = os.path.join(dir, 'train.jsonl')
        X_train = Dataset.from_json(train_path) if os.path.exists(train_path) else None

        dev_path = os.path.join(dir, 'validation.jsonl')
        X_dev = Dataset.from_json(dev_path) if os.path.exists(dev_path) else None


    if preprocess_test:

        test_path = os.path.join(dir, 'input.jsonl')
        if os.path.exists(test_path):

            X_test = Dataset.from_pandas(
                squad_format(test_path)
            )
            X_test = X_test.map(from_squad, batched=True)

        else: X_test = None

    else:

        test_path = os.path.join(dir, 'input.jsonl')
        X_test = Dataset.from_json(test_path) if os.path.exists(test_path) else None
        

    return X_train, X_dev, X_test