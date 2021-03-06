from random import shuffle
from transformers.integrations import hp_params
import data_loader
import torch
import argparse
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from preprocessor import Preprocessor
from transformers import BertTokenizerFast, BertForSequenceClassification, AlbertTokenizerFast, AlbertForSequenceClassification
from transformers import logging as hf_logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
from linear_repeat import LABELS
from bert_parallel_token_classification import BertForParallelTokenClassification
from albert_parallel_token_classification import AlbertForParallelTokenClassification
from collections import defaultdict
import os
import time
import json

parser = argparse.ArgumentParser(description='Train a network to identify arms deals.')
parser.add_argument('data', type=str, help="The dataset directory.")
parser.add_argument('-l', '--load', default=None, type=str, help="Load the model from a file.")
parser.add_argument('--task', type=str, default="token", help="The thing to classify.")
parser.add_argument('--classifier', type=str, default="albert", help="Classify with bert or albert.")
parser.add_argument('--max_epochs', type=int, default=100, help="Max epochs.")
parser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
parser.add_argument('--print', action='store_true', help="Print classifications after training.")
parser.add_argument('--print_train', action='store_true', help="Print classifications of training data after training.")
parser.add_argument('--print_test', action='store_true', help="Print classifications of testing data after training.")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training.")
parser.add_argument('--small_data', action='store_true', help="Only use a small part of the dataset for debugging.")
parser.add_argument('--train_portion', type=float, default=0.95, help="Proportion of data to use for training (the rest is for validation).")
parser.add_argument('--max_tokens', type=int, default=128, help="Max length of a tokenization.")
parser.add_argument('--dataloader_workers', type=int, default=16, help="Number of dataloader workers.")
parser.add_argument('--tune', type=str, help="Run fine-tuning algorithm and save to a file with filename.")
parser.add_argument('--resume', type=int, default=0, help="Starting point for fine tuning.")
parser.add_argument('--test', action='store_true', help="Run model on test data.")
parser.add_argument('--split', type=str, default="fixed", help="Split into chunks or fixed")
args = parser.parse_args()

assert args.split in ("fixed", "chunks")
assert args.task in ("sequence", "token")

def forward_wrapper(encoder, text, labels):
  output = encoder(text, labels=labels)
  if labels != None:
    # Output loss and probabilities
    loss, probs = output[:2]
    return loss, probs
  # Output only probabilities
  probs = output[0]
  return probs

class BERT_token(nn.Module):
  options_name = "bert-base-cased"
  def __init__(self):
    super(BERT_token, self).__init__()
    self.encoder = BertForParallelTokenClassification.from_pretrained(self.options_name)
  def forward(self, text, labels=None):
    return forward_wrapper(self.encoder, text, labels)

class BERT_sequence(nn.Module):
  options_name = "bert-base-cased"
  def __init__(self):
    super(BERT_sequence, self).__init__()
    self.encoder = BertForSequenceClassification.from_pretrained(self.options_name)
  def forward(self, text, labels=None):
    return forward_wrapper(self.encoder, text, labels)

class ALBERT_token(nn.Module):
  options_name = "albert-base-v2"
  def __init__(self):
    super(ALBERT_token, self).__init__()
    self.encoder = AlbertForParallelTokenClassification.from_pretrained(self.options_name)
  def forward(self, text, labels=None):
    return forward_wrapper(self.encoder, text, labels)

class ALBERT_sequence(nn.Module):
  options_name = "albert-base-v2"
  def __init__(self):
    super(ALBERT_sequence, self).__init__()
    self.encoder = AlbertForSequenceClassification.from_pretrained(self.options_name)
  def forward(self, text, labels=None):
    return forward_wrapper(self.encoder, text, labels)

class TextDataset(Dataset):

  def __init__(self, texts, preprocessor, task='token'):
    self.datapoints = []
    for text in texts:
      if task == 'token':
        x, y = preprocessor.labels_multiple(text, LABELS)
      else:
        x, y = preprocessor.binary(text)
      self.datapoints.append((x, y))

  def __len__(self):
    return len(self.datapoints)

  def __getitem__(self, idx):
    return self.datapoints[idx]

class LitModule(pl.LightningModule):

  def get_model(self):
    if self.hparams.classifier == 'bert':
      return BERT_token() if self.hparams.task == 'token' else BERT_sequence()
    else:
      return ALBERT_token() if self.hparams.task == 'token' else ALBERT_sequence()

  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters(config)
    assert self.hparams.task in ('token', 'sequence')
    assert self.hparams.classifier in ('bert', 'albert')
    hf_logging.set_verbosity_error()
    self.model = self.get_model()
    hf_logging.set_verbosity_warning()

    self.best_hp_metric = -1
    metrics = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Precision(num_classes=2, ignore_index=0, mdmc_average='global'),
        pl.metrics.Recall(num_classes=2, ignore_index=0, mdmc_average='global')
    ])
    self.training_metrics = metrics.clone()
    self.validation_metrics = metrics.clone()
    self.test_metrics = metrics.clone()

  def forward(self, x):
    return self.probs_to_preds(self.model(x))

  def training_step(self, batch, batch_idx):
    x, y = batch
    loss, probs = self.model(x, y)
    preds = self.probs_to_preds(probs)
    self.training_metrics(preds, y)
    return loss
  
  def training_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.training_metrics.compute())
    self.training_metrics.reset()
    metrics["Loss"] = torch.mean(torch.Tensor([output['loss'].item() for output in outputs]))
    self.log_metrics("Training", metrics)
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    loss, probs = self.model(x, y)
    preds = self.probs_to_preds(probs)
    self.validation_metrics(preds, y)
    return loss
  
  def validation_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.validation_metrics.compute())
    self.validation_metrics.reset()
    metrics["Loss"] = torch.mean(torch.Tensor(outputs))
    self.log_metrics("Validation", metrics)
    self.save_hp_metric(metrics['F1'])
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    loss, probs = self.model(x, y)
    preds = self.probs_to_preds(probs)
    self.test_metrics(preds, y)
    return loss

  def test_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.test_metrics.compute())
    self.test_metrics.reset()
    metrics["Loss"] = torch.mean(torch.Tensor(outputs))
    self.log_metrics("Test", metrics)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    return optimizer
  
  def prepare_data(self):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if self.hparams.classifier == 'bert':
      tokenizer = BertTokenizerFast.from_pretrained(self.model.options_name)
    else:
      tokenizer = AlbertTokenizerFast.from_pretrained(self.model.options_name)
    self.preprocessor = Preprocessor(tokenizer, self.hparams.max_tokens)

    def get_splits(dataset, split):
      return (dataset.split_fixed(shuffle=True) \
          if split == 'fixed' else \
          dataset.split_chunks(shuffle=True)).texts

    texts = get_splits(data_loader.DataSet.load_json2(args.data), self.hparams.split)
    if self.hparams.small_data:
      texts = texts[:20]
    self.dataset = TextDataset(texts, self.preprocessor, self.hparams.task)

    test_texts = get_splits(data_loader.DataSet.load_json2("test_" + args.data), self.hparams.split)
    self.test_dataset = TextDataset(test_texts, self.preprocessor, self.hparams.task)

    eval_start = int(len(self.dataset)*self.hparams.train_portion)
    self.train_sampler = SubsetRandomSampler(range(0, eval_start))
    self.val_sampler = SubsetRandomSampler(range(eval_start, len(self.dataset)))

  def train_dataloader(self):
    global num_workers
    return DataLoader(
        self.dataset,
        num_workers=num_workers, 
        batch_size=self.hparams.batch_size, 
        sampler=self.train_sampler)
  
  def val_dataloader(self):
    global num_workers
    return DataLoader(
        self.dataset,
        num_workers=num_workers, 
        batch_size=self.hparams.batch_size, 
        sampler=self.val_sampler)
  
  def test_dataloader(self):
    global num_workers
    return DataLoader(
        self.test_dataset,
        num_workers=num_workers, 
        batch_size=self.hparams.batch_size)

  def probs_to_preds(self, probabilities):
    return torch.argmax(probabilities, dim=len(probabilities.size())-1)
  
  def log_metrics(self, headline, metrics, logger=True):
    for metric in metrics:
      name = "{}{}".format(headline, metric)
      self.log(name, metrics[metric], logger=logger)
  
  def extend_metrics(self, metrics):
    precision, recall = metrics['Precision'], metrics['Recall']
    metrics['F1'] = 2*(precision*recall)/(precision + recall) if precision + recall else 0
    return metrics
  
  def save_hp_metric(self, metric):
    self.log("hp_metric", metric, logger=True)
    self.best_hp_metric = max(self.best_hp_metric, metric)
  
  def print_batch(self, batch):
    self.model.eval()
    stats = defaultdict(lambda: (0,)*4)
    with torch.no_grad():
      x, y = batch
      y_pred = self.forward(x)

      for x_text, y_text, y_pred_text in zip(x, y, y_pred):
        if self.hparams.task == 'token':
          for label, y_label, y_pred_label in zip(LABELS, 
              torch.transpose(y_text, 0, 1),
              torch.transpose(y_pred_text, 0, 1)):
            print("\033[7m", label, "\033[0m")
            batch_stats = self.preprocessor.print_labels(x_text, y_label, y_pred_label)
            stats[label] = tuple(sum(x) for x in zip(batch_stats, stats[label]))
        else:
          self.preprocessor.print_sequence(x_text, y_text, y_pred_text)
        print()
      print()
    return stats

num_workers = args.dataloader_workers

default_config = {
    'lr':args.lr, 
    'batch_size':args.batch_size,
    'task':args.task, 
    'classifier':args.classifier,
    'small_data':args.small_data,
    'train_portion':args.train_portion,
    'split':args.split,
    'max_tokens':args.max_tokens}

if args.tune:
  lr_values = (1e-4, 1e-5, 1e-6)
  bs_values = (2, 4, 8)
  all_configs = [(lr, bs) for lr in lr_values for bs in bs_values]
  output = {
    'results':[],
    'default_config':default_config,
    'max_epochs':args.max_epochs
  }
  results = []
  for lr, bs in all_configs[args.resume:]:
    print("Trying learning rate {}, batch size {}.".format(lr, bs))

    trainer = pl.Trainer(
      gpus=1 if args.gpu else 0,
      default_root_dir="lightning",
      max_epochs=args.max_epochs)

    config = default_config.copy()
    config['lr'] = lr
    config['batch_size'] = bs
    lit_module = LitModule(config)

    start = time.time()
    trainer.fit(lit_module)
    end = time.time()

    output['results'].append({
      'lr': lr,
      'bs': bs,
      'hp_metric': float(lit_module.best_hp_metric),
      'time': end - start,
      'version': lit_module.logger.version
    })
    del lit_module
    del trainer
    torch.cuda.empty_cache()
    filename = args.tune + ".json"
    sleep_time = 3
    with open(filename, 'w') as file:
      json.dump(output, file, indent=2)
    print("Test complete, saving results to {}. Sleeping for {} seconds.".format(filename, sleep_time))
    print("Last results:", output['results'][-1])
    time.sleep(sleep_time) # Sleep to allow time for garbage collection
  exit()



kwargs = {}
callbacks = []

if args.load:
  lit_module = LitModule.load_from_checkpoint(args.load)
  kwargs['resume_from_checkpoint'] = args.load
else:
  lit_module = LitModule(default_config)


callbacks.append(ModelCheckpoint(
      monitor='hp_metric',
      filename='save-{epoch:02d}-{hp_metric:.3f}',
      save_top_k=1,
      save_last=True,
      mode='max',
  )
)

trainer = pl.Trainer(
    gpus=1 if args.gpu else 0,
    default_root_dir="lightning",
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    **kwargs)


trainer.fit(lit_module)
if args.print:
  for batch in lit_module.val_dataloader():
    lit_module.print_batch(batch)
if args.print_train:
  for batch in lit_module.train_dataloader():
    lit_module.print_batch(batch)

print("Best model score: ", trainer.checkpoint_callback.best_model_score)
print("Best model path: ", trainer.checkpoint_callback.best_model_path)

if args.test:
  print("Proceed with testing? (y/n)")
  if input()[0].lower() == 'y':
    path = trainer.checkpoint_callback.best_model_path
    lit_module = LitModule.load_from_checkpoint(path)
    trainer.test(lit_module)

if args.print_test:
  def calculate_metrics(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 / (1/precision + 1/recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1, accuracy

  path = trainer.checkpoint_callback.best_model_path
  lit_module = LitModule.load_from_checkpoint(path)
  lit_module.prepare_data()
  stats = defaultdict(lambda: (0,)*4)
  for batch in lit_module.test_dataloader():
    stats_here = lit_module.print_batch(batch)
    for label in stats_here:
      stats[label] = tuple(sum(x) for x in zip(stats_here[label], stats[label]))
      precision, recall, f1, accuracy = calculate_metrics(*stats[label])
      print("{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\\hline".format(label, f1, precision, recall, accuracy))

# My experiments:
# Sequence fixed: python3 training_lightning.py --max_epochs 100 --gpu --load lightning/lightning_logs/version_7/checkpoints/last.ckpt --print_test data.json
# Token fixed: python3 training_lightning.py --max_epochs 200 --gpu --load lightning/lightning_logs/version_34/checkpoints/last.ckpt --print_test data.json
# Sequence chunks: python3 training_lightning.py --max_epochs 38 --gpu --load lightning/lightning_logs/version_15/checkpoints/last.ckpt --print_test data.json
# Token chunks: python3 training_lightning.py --max_epochs 42 --gpu --load lightning/lightning_logs/version_35/checkpoints/last.ckpt --print_test data.json