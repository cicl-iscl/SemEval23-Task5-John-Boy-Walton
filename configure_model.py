from transformers import TrainingArguments, AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding
from dataclasses import dataclass, field, asdict
from datasets import Dataset
from collections import defaultdict

from qa_trainer import QATrainer
from metrics import bleu
from postprocess import postprocess


# https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments
@dataclass
class CTrainingArguments:

    '''
    Allows custom TrainingArguments default values.
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
        default=0.001
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

    @property
    def configured(self):
        return TrainingArguments(**asdict(self))


# https://huggingface.co/deepset/roberta-base-squad2
DEFAULT_MODEL_NAME = 'deepset/roberta-base-squad2'
# https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.Trainer
@dataclass(init=True)
class CTrainer:
    
    '''
    Allows custom Trainer default values.
    '''

    model: AutoModelForQuestionAnswering = field(
        default=AutoModelForQuestionAnswering.from_pretrained(DEFAULT_MODEL_NAME)
    )
    args: TrainingArguments = field(
        default=CTrainingArguments().configured
    )
    train_dataset: Dataset = field(
        default=None
    )
    eval_dataset: Dataset = field(
        default=None
    )
    tokenizer: AutoTokenizer = field(
        default=AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    )
    data_collator: DataCollatorWithPadding = field(
        default=DataCollatorWithPadding(tokenizer=tokenizer.default)
    )
    compute_metrics: callable = field(
        default=bleu
    )
    post_process_function: callable = field(
        default=postprocess
    )
    post_process_kwargs: defaultdict = field(
        default_factory=lambda: {
            'n_best_size':       15,
            'max_answer_length': 20,
            'top_k': 3,
            'output_dir': 'flow/predictions'
        }
    )

    @property
    def asdict_(self): # can't use asdict(self): throws an error
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__.keys()
        }

    @property
    def configured(self):
        return QATrainer(**self.asdict_)
