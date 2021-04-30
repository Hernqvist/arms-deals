import data_loader
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from preprocessor import Preprocessor
from transformers import BertTokenizerFast
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from parallel_token_classification import BertForParallelTokenClassification, LABELS

parser = argparse.ArgumentParser(description='Train a network to identify arms deals.')
parser.add_argument('data', type=str, help="The dataset directory.")
parser.add_argument('-l', '--load', default=None, type=str, help="Load the model from a file.")
parser.add_argument('-s', '--save', default=None, type=str, help="Save the model under this name between epochs.")
parser.add_argument('--print_evals', action='store_true', help="Print classifications of evaluation set.")
parser.add_argument('--eval_on_start', action='store_true', help="Evaluate before any training.")
args = parser.parse_args()

class BERT(nn.Module):
  max_length = 128
  options_name = "bert-base-cased"

  def __init__(self):
    super(BERT, self).__init__()
    self.encoder = BertForParallelTokenClassification.from_pretrained(self.options_name)

  def forward(self, text, labels=None):
    output = self.encoder(text, labels=labels)
    if labels != None:
      # Output lo ss and probabilities
      loss, probs = output[:2]
      return loss, probs
    # Output only probabilities
    probs = output[0]
    return probs

class TextDataset(Dataset):

  def __init__(self, texts, preprocessor):
    self.datapoints = []
    for text in texts:
      x, y = preprocessor.labels_multiple(text, LABELS)
      self.datapoints.append((x, y))

  def __len__(self):
    return len(self.datapoints)

  def __getitem__(self, idx):
    return self.datapoints[idx]

class LightningModel(pl.LightningModule):

  def __init__(self, config):
    super().__init__()
    self.model = BERT()
    self.config = config
    self.save_hyperparameters(config)
    self.best_hp_metric = -1
    metrics = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Precision(num_classes=2, ignore_index=0, mdmc_average='global'),
        pl.metrics.Recall(num_classes=2, ignore_index=0, mdmc_average='global')
    ])
    self.training_metrics = metrics.clone()
    self.validation_metrics = metrics.clone()

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
    metrics["Loss"] = torch.mean(torch.Tensor([output['loss'].item() for output in outputs]))
    self.log_metrics("Training", metrics)
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    preds = self.forward(x)
    self.validation_metrics(preds, y)
  
  def validation_epoch_end(self, outputs):
    metrics = self.extend_metrics(self.validation_metrics.compute())
    self.log_metrics("Validation", metrics)
    self.save_hp_metric(metrics['F1'])

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
    return optimizer

  def train_dataloader(self):
    global num_workers
    return DataLoader(
        dataset,
        num_workers=num_workers, 
        batch_size=self.config['batch_size'], 
        sampler=train_sampler)
  
  def val_dataloader(self):
    global num_workers
    return DataLoader(
        dataset,
        num_workers=num_workers, 
        batch_size=self.config['batch_size'], 
        sampler=train_sampler)

  def probs_to_preds(self, probabilities):
    return torch.argmax(probabilities, dim=3)
  
  def log_metrics(self, headline, metrics):
    for metric in metrics:
      name = "{}{}".format(headline, metric)
      self.log(name, metrics[metric], logger=True)
  
  def extend_metrics(self, metrics):
    precision, recall = metrics['Precision'], metrics['Recall']
    metrics['F1'] = 2*(precision*recall)/(precision + recall)
    return metrics
  
  def save_hp_metric(self, metric):
    self.log("hp_metric", metric)
    self.best_hp_metric = max(self.best_hp_metric, metric)


texts = data_loader.DataSet.load_json2(args.data).split_chunks().texts[:6]
tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
preprocessor = Preprocessor(tokenizer, BERT.max_length)
dataset = TextDataset(texts, preprocessor)
split_index = (len(dataset)*2)//3
train_sampler = SubsetRandomSampler(range(0, split_index))
val_sampler = SubsetRandomSampler(range(split_index, len(dataset)))

batch_size = 2
num_workers = 0

lightning_model = LightningModel({'lr':5e-5, 'batch_size':2})
trainer = pl.Trainer()
trainer.fit(lightning_model)