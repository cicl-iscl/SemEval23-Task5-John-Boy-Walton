#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 0
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import TrainingArguments, Trainer, \
                         AutoModelForQuestionAnswering, AutoModelForSequenceClassification, \
                         AutoTokenizer, DataCollatorWithPadding
import torch
from dataclasses import dataclass, field
from datasets import Dataset

from utils.metrics import bleu, accuracy
from utils.postprocess import postprocess_qa, postprocess_classify

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_MODEL_NAME = './models/distilbert-base-uncased'
# DEFAULT_MODEL_NAME = '/WebSemble/models/distilbert-base-uncased'


#region BaseModels

@dataclass
class WebBase:

    '''
    Base class for casting custom TrainingArguments and Trainers.
    '''
    kwargs: dict = field(
        default_factory=lambda: {}
    )

    @property
    def asdict_(self): # can't use asdict(self) as we must handle **kwargs
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__.keys()
            if not field == 'kwargs'
        }

    @property
    def configured(self):
        raise NotImplementedError('Configure models in your derived class.')


@dataclass
class WebTrainingArgs(WebBase):

    @property
    def configured(self):
        return TrainingArguments(**self.asdict_, **self.kwargs)


@dataclass
class WebTrainer(WebBase):

    @property
    def configured(self):
        return PostTrainer(**self.asdict_, **self.kwargs)

#endregion


#region TrainingArguments

@dataclass
class WebQATrainingArguments(WebTrainingArgs):

    '''
    Allows custom TrainingArguments default values for QA.
    '''

    output_dir: str = field(
        default='flow/checkpoints',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    log_level: str = field(
        default='info',
        metadata={'help': '‘debug’, ‘info’, ‘warning’, ‘error’, ‘critical’ or ‘passive’.'}
    )
    logging_dir: str = field(
        default='flow/log'
    )
    logging_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    logging_steps: int = field(
        default=25,
        metadata={'help': 'Number of update steps between two logs.'}
    )
    do_train: bool = field(
        default=True
    )
    do_eval: bool = field(
        default=True
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    eval_steps: int = field(
        default=250,
        metadata={'help': 'Number of update steps between two evaluations.'}
    )
    do_predict: bool = field(
        default=False
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    save_steps: int = field(
        default=2500,
        metadata={'help': 'Number of updates steps before two checkpoint saves.'}
    )
    load_best_model_at_end: bool = field(
        default=True
    )
    metric_for_best_model: str = field(
        default='eval_bleu'
    )
    greater_is_better: bool = field(
        default=True,
        metadata={'help': 'If better models should have a greater metric or not.'}
    )
    per_device_train_batch_size: int = field(
        default=8
    )
    per_device_eval_batch_size: int = field(
        default=8
    )
    learning_rate: float = field(
        default=2e-5
    )
    lr_scheduler_type: str = field(
        default='constant'
    )
    optim: str = field(
        default='adamw_hf'
    )
    num_train_epochs: int = field(
        default=5
    )
    max_steps: int = field(
        default=-1
    )
    group_by_length: bool = field(
        default=True,
        metadata={'help': 'Whether or not to group together samples of roughly the same length in the training dataset.'}
    )
    report_to: str or list = field(
        default='none',
        metadata={'help': 'The list of integrations to report the results and logs to.'}
    )
    kwargs: dict = field(
        default_factory=lambda: {}
    )


@dataclass
class WebClassificationTrainingArguments(WebTrainingArgs):

    '''
    Allows custom TrainingArguments default values for Text Classification.
    '''
    output_dir: str = field(
        default='flow/checkpoints',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    log_level: str = field(
        default='info',
        metadata={'help': '‘debug’, ‘info’, ‘warning’, ‘error’, ‘critical’ or ‘passive’.'}
    )
    logging_dir: str = field(
        default='flow/log'
    )
    logging_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    logging_steps: int = field(
        default=25,
        metadata={'help': 'Number of update steps between two logs.'}
    )
    do_train: bool = field(
        default=True
    )
    do_eval: bool = field(
        default=True
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    eval_steps: int = field(
        default=50,
        metadata={'help': 'Number of update steps between two evaluations.'}
    )
    do_predict: bool = field(
        default=False
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': '‘no’, ‘epoch’ or ‘steps’.'}
    )
    save_steps: int = field(
        default=100,
        metadata={'help': 'Number of updates steps before two checkpoint saves.'}
    )
    load_best_model_at_end: bool = field(
        default=True
    )
    metric_for_best_model: str = field(
        default='eval_accuracy'
    )
    greater_is_better: bool = field(
        default=True,
        metadata={'help': 'If better models should have a greater metric or not.'}
    )
    per_device_train_batch_size: int = field(
        default=8
    )
    per_device_eval_batch_size: int = field(
        default=8
    )
    learning_rate: float = field(
        default=2e-5
    )
    lr_scheduler_type: str = field(
        default='constant'
    )
    optim: str = field(
        default='adamw_hf'
    )
    num_train_epochs: int = field(
        default=2.5
    )
    max_steps: int = field(
        default=-1
    )
    group_by_length: bool = field(
        default=False,
        metadata={'help': 'Whether or not to group together samples of roughly the same length in the training dataset.'}
    )
    report_to: str or list = field(
        default='none',
        metadata={'help': 'The list of integrations to report the results and logs to.'}
    )
    kwargs: dict = field(
        default_factory=lambda: {}
    )

#endregion


#region Trainers

class PostTrainer(Trainer):

    def __init__(self, post_process_function=None, post_process_kwargs=None, **kwargs):
        super(PostTrainer, self).__init__(**kwargs)
        self.post_process_function = post_process_function
        self.post_process_kwargs = post_process_kwargs

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # temporarily disable metric computation, we will do it in the loop here
        # because evaluation_loop() wants compute metrics right away
        # if self.compute_metrics is not None but ours requires additional args
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        output = self.evaluation_loop(
            eval_dataloader,
            description='Evaluation',
            ignore_keys=ignore_keys
        )
        self.compute_metrics = compute_metrics

        eval_preds = self.post_process_function(eval_dataset, output.predictions, **self.post_process_kwargs)
        metrics = self.compute_metrics(eval_preds)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        log_history = self.state.log_history # dict      
        if len(log_history):
            all_metrics = {**log_history[-1], **metrics}
            print(all_metrics)
            # self.save_metrics('eval', all_metrics)
        return metrics


# https://huggingface.co/deepset/roberta-base-squad2
DEFAULT_QA_MODEL_NAME = './models/roberta-base-squad2'
# DEFAULT_QA_MODEL_NAME = '/WebSemble/models/roberta-base-squad2'
@dataclass
class WebQATrainer(WebTrainer):
    
    '''
    Allows custom Trainer default values for QA.
    '''

    model: AutoModelForQuestionAnswering = field(
        default=AutoModelForQuestionAnswering.from_pretrained(DEFAULT_QA_MODEL_NAME).to(DEVICE)
    )
    args: TrainingArguments = field(
        default=WebQATrainingArguments().configured
    )
    train_dataset: Dataset = field(
        default=None
    )
    eval_dataset: Dataset = field(
        default=None
    )
    tokenizer: AutoTokenizer = field(
        default=AutoTokenizer.from_pretrained(DEFAULT_QA_MODEL_NAME)
    )
    data_collator: DataCollatorWithPadding = field(
        default=DataCollatorWithPadding(tokenizer=tokenizer.default)
    )
    compute_metrics: callable = field(
        default=bleu
    )
    post_process_function: callable = field(
        default=postprocess_qa
    )
    post_process_kwargs: dict = field(
        default_factory=lambda: {}
    )
    kwargs: dict = field(
        default_factory=lambda: {}
    )


# https://huggingface.co/google/pegasus-xsum
DEFAULT_S2S_MODEL_NAME = './models/pegasus-xsum'
# DEFAULT_S2S_MODEL_NAME = '/WebSemble/models/pegasus-xsum'


# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
DEFAULT_CLASSIFICATION_MODEL_NAME = './models/distilbert-base-uncased'
# DEFAULT_CLASSIFICATION_MODEL_NAME = '/WebSemble/models/distilbert-base-uncased'
@dataclass
class WebClassificationTrainer(WebTrainer):

    '''
    Allows custom Trainer default values for Text Classification.
    '''

    model: AutoModelForSequenceClassification = field(
        default=AutoModelForSequenceClassification.from_pretrained(DEFAULT_CLASSIFICATION_MODEL_NAME).to(DEVICE)
    )
    args: TrainingArguments = field(
        default=WebClassificationTrainingArguments().configured
    )
    train_dataset: Dataset = field(
        default=None
    )
    eval_dataset: Dataset = field(
        default=None
    )
    tokenizer: AutoTokenizer = field(
        default=AutoTokenizer.from_pretrained(DEFAULT_CLASSIFICATION_MODEL_NAME)
    )
    data_collator: DataCollatorWithPadding = field(
        default=DataCollatorWithPadding(tokenizer=tokenizer.default)
    )
    compute_metrics: callable = field(
        default=accuracy
    )
    post_process_function: callable = field(
        default=postprocess_classify
    )
    post_process_kwargs: dict = field(
        default_factory=lambda: {}
    )
    kwargs: dict = field(
        default_factory=lambda: {}
    )

#endregion