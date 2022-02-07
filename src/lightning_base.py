import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import LearningRateLogger

from modeling_de import DeBartForConditionalGeneration
from modeling_def import DefBartForConditionalGeneration
from modeling_re import ReBartForConditionalGeneration
from modeling_kg import KgBartForConditionalGeneration
from modeling_kd import KdBartForConditionalGeneration

# from modeling_t5_de import DeT5ForConditionalGeneration
from modeling_t5_def import DefT5ForConditionalGeneration
from modeling_t5_re import ReT5ForConditionalGeneration
from modeling_t5_kg import KgT5ForConditionalGeneration
from modeling_t5_kd import KdT5ForConditionalGeneration

from transformers import (
    AdamW,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    T5Tokenizer,
    T5Config
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

logger = logging.getLogger(__name__)

BART_MODEL_MODES = {
    "graph2text": BartForConditionalGeneration,
    "de2text": DeBartForConditionalGeneration,
    "def2text": DefBartForConditionalGeneration,
    "re2text": ReBartForConditionalGeneration,
    "kg2text": KgBartForConditionalGeneration,
    "kgm2text": KgBartForConditionalGeneration,
    "kd2text": KdBartForConditionalGeneration
}

T5_MODEL_MODES = {
    "graph2text": T5ForConditionalGeneration,
    # "de2text": DeT5ForConditionalGeneration,
    "def2text": DefT5ForConditionalGeneration,
    "re2text": ReT5ForConditionalGeneration,
    "kg2text": KgT5ForConditionalGeneration,
    "kgm2text": KgT5ForConditionalGeneration,
    "kd2text": KdT5ForConditionalGeneration
}

# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

        self.save_hyperparameters(hparams)
        self.step_count = -2
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None

        if 'bart' in self.hparams.model_name_or_path.lower():
            self.plm = 'bart'
        elif 't5' in self.hparams.model_name_or_path.lower():
            self.plm = 't5'

        assert config is None
        if self.plm == 'bart':
            self.config = BartConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        elif self.plm == 't5':
            self.config = T5Config.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )

        self.config.return_dict = True
        self.config.output_hidden_states = True
        if not hasattr(self.config, 'N'):
            self.config.N = self.hparams.N
        if not hasattr(self.config, 'not_concat'):
            self.config.not_concat = self.hparams.not_concat
        if not hasattr(self.config, 'use_unk'):
            self.config.use_unk = self.hparams.use_unk
        if not hasattr(self.config, 'pooling_type'):
            self.config.pooling_type = self.hparams.pooling_type

        self.config.add_classifier = False

        dropout = getattr(self.hparams, "dropout", None)
        if dropout:
            self.config.dropout = dropout
            self.config.attention_dropout = dropout
            self.config.activation_dropout = dropout

        # extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        # for p in extra_model_params:
        #     if getattr(self.hparams, p, None):
        #         assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
        #         setattr(self.config, p, getattr(self.hparams, p))

        assert tokenizer is None
        if self.plm == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
        elif self.plm == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )

        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        rank_zero_info('We have added %s tokens.', num_added_toks)

        self.mode = self.hparams.task
        if self.mode == 're2text' and self.hparams.loss_epoch == 0:
            self.config.add_classifier = True

        if self.plm == 'bart':
            self.model_type = BART_MODEL_MODES[self.mode]
        elif self.plm == 't5':
            self.model_type = T5_MODEL_MODES[self.mode]

        if self.hparams.baseline:
            self.model = self.model_type(self.config)
            rank_zero_info('Train the %s model with %s task from scratch.' % (self.plm, self.mode))
            if self.mode == 'kg2text' or self.mode == 'kd2text':
                self.load_kg_embedding()
        else:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir,
            )
            rank_zero_info('Train the %s model with %s task by fine-tune.' % (self.plm, self.mode))
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = get_schedule_func(
                self.opt, num_warmup_steps=self.hparams.warmup_steps
            )
        else:
            scheduler = get_schedule_func(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
            )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    @property
    def total_steps(self) -> int:
        # print('self.hparams.gpus', self.hparams.gpus)
        # print('self.hparams.accumulate_grad_batches', self.hparams.accumulate_grad_batches)
        # print('self.train_loader.dataset', self.train_loader.dataset)
        # print('self.hparams.max_epochs', self.hparams.max_epochs)
        # print('self.hparams.train_batch_size', self.hparams.train_batch_size)
        # exit()
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        # print('self.hparams.gpus', self.hparams.gpus)
        # print('self.hparams.accumulate_grad_batches', self.hparams.accumulate_grad_batches)
        # print('self.train_loader.dataset', self.train_loader.dataset)
        # print('self.hparams.max_epochs', self.hparams.max_epochs)
        # print('self.hparams.train_batch_size', self.hparams.train_batch_size)
        # exit()
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return dataset_size / effective_batch_size

    def load_kg_embedding(self):
        model_weights = torch.load(self.hparams.model_name_or_path + 'pytorch_model_kg.bin')
        self.model.model.encoder.kg_embeddings.weight.data = model_weights['model.encoder.kg_embeddings.weight']
        rank_zero_info('Load kg embedding.')


    def setup(self, mode):
        self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    def get_progress_bar_dict(self):
        lrs = self.trainer.lr_logger.lrs['lr-AdamW/pg1'][-1]
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss), "lr": lrs}
        return tqdm_dict

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument("--adafactor", action="store_true")


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        rank_zero_info(trainer.logger)
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir) -> None:
    #  TODO(SS): allow all pl args? parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )


def generic_train(
    model: BaseTransformer,
    args: argparse.Namespace,
    early_stopping_callback=False,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches

    lr_logger = LearningRateLogger(logging_interval='step')

    #         deterministic=True,
    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary='full',
        callbacks=[logging_callback, lr_logger],
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        num_sanity_val_steps=4,
        progress_bar_refresh_rate=100,
        **train_params,
    )

    trainer.lr_logger = lr_logger

    return trainer
