import data_loader
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from preprocessor import Preprocessor
from transformers import BertTokenizerFast, BertForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F
from parallel_token_classification import BertForParallelTokenClassification, LABELS

parser = argparse.ArgumentParser(description='Train a network to identify arms deals.')
parser.add_argument('data', type=str, help="The dataset directory.")
parser.add_argument('-l', '--load', default=None, type=str, help="Load the model from a file.")
parser.add_argument('-s', '--save', action='store_true', help="Save the model between epochs.")
parser.add_argument('--task', type=str, default="token", help="The thing to classify.")
parser.add_argument('--max_epochs', type=int, default=100, help="Max epochs.")
parser.add_argument('--print', action='store_true', help="Print classifications after training.")
#parser.add_argument('--eval_on_start', action='store_true', help="Evaluate before any training.")
args = parser.parse_args()

class BERT_token(nn.Module):
  options_name = "bert-base-cased"
  max_length = 128

  def __init__(self):
    super(BERT_token, self).__init__()
    self.encoder = BertForParallelTokenClassification.from_pretrained(self.options_name)

  def forward(self, text, labels=None):
    output = self.encoder(text, labels=labels)
    if labels != None:
      # Output loss and probabilities
      loss, probs = output[:2]
      return loss, probs
    # Output only probabilities
    probs = output[0]
    return probs

class BERT_sequence(nn.Module):
  options_name = "bert-base-cased"
  max_length = 128

  def __init__(self):
    super(BERT_sequence, self).__init__()
    self.encoder = BertForSequenceClassification.from_pretrained(self.options_name)

  def forward(self, text, labels=None):
    output = self.encoder(text, labels=labels)
    if labels != None:
      # Output loss and probabilities
      loss, probs = output[:2]
      return loss, probs
    # Output only probabilities
    probs = output[0]
    return probs

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
    return BERT_token() if self.hparams.task == 'token' else BERT_sequence()

  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters(config)
    assert self.hparams.task in ('token', 'sequence')
    self.model = self.get_model()

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
    # in lightning, forward defines the prediction/inference actions
    return self.probs_to_preds(self.model(x))

  def training_step(self, batch, batch_idx):
    # training_step defines the train loop. It is independent of forward
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
    preds = self.forward(x)
    self.validation_metrics(preds, y)
  
  def validation_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.validation_metrics.compute())
    self.validation_metrics.reset()
    self.log_metrics("Validation", metrics)
    self.save_hp_metric(metrics['F1'])
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    preds = self.forward(x)
    self.validation_metrics(preds, y)
  
  def test_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.test_metrics.compute())
    self.test_metrics.reset()
    self.log_metrics("Test", metrics)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    return optimizer
  
  def prepare_data(self):
    texts = data_loader.DataSet.load_json2(args.data).split_chunks().texts[:20]
    tokenizer = BertTokenizerFast.from_pretrained(self.model.options_name)
    self.preprocessor = Preprocessor(tokenizer, self.model.max_length)
    self.dataset = TextDataset(texts, self.preprocessor, self.hparams.task)


    eval_start = int(len(self.dataset)*0.7)
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
        sampler=self.train_sampler)
  
  def test_dataloader(self):
    global num_workers
    return DataLoader(
        self.dataset,
        num_workers=num_workers, 
        batch_size=self.hparams.batch_size, 
        sampler=self.test_sampler)

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
    self.log("hp_metric", metric)
    self.best_hp_metric = max(self.best_hp_metric, metric)
  
  def print_batch(self, batch):
    self.model.eval()
    with torch.no_grad():
      x, y = batch
      y_pred = self.forward(x)

      for x_text, y_text, y_pred_text in zip(x, y, y_pred):
        print()
        if self.hparams.task == 'token':
          for label, y_label, y_pred_label in zip(LABELS, 
              torch.transpose(y_text, 0, 1),
              torch.transpose(y_pred_text, 0, 1)):
            print("\033[7m", label, "\033[0m")
            self.preprocessor.print_labels(x_text, y_label, y_pred_label)
        else:
          self.preprocessor.print_sequence(x, y_text, y_pred_text)
      print()

num_workers = 0
kwargs = {}
callbacks = []

if args.load:
  lit_module = LitModule.load_from_checkpoint(args.load)
  kwargs['resume_from_checkpoint'] = args.load
else:
  lit_module = LitModule({'lr':5e-4, 'batch_size':2, 'task':args.task})

if args.save:
  callbacks.append(ModelCheckpoint(
        monitor='hp_metric',
        filename='save-{epoch:02d}-{hp_metric:.3f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
  )

trainer = pl.Trainer(
    default_root_dir="lightning",
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    **kwargs)


trainer.fit(lit_module)
if args.print:
  for batch in lit_module.train_dataloader():
    lit_module.print_batch(batch)
#trainer.test(lit_module)