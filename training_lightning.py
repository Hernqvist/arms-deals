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
      # Output loss and probabilities
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

  def probs_to_preds(self, probabilities):
    return torch.argmax(probabilities, dim=3)

  def __init__(self):
    super().__init__()
    self.model = BERT()
    metrics = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Precision(num_classes=2, ignore_index=0, mdmc_average='global')
    ])
    self.train_metrics = pl.metrics.Precision(num_classes=2, ignore_index=0, mdmc_average='global')
    self.valid_metrics = pl.metrics.Precision(num_classes=2, ignore_index=0, mdmc_average='global')

  def forward(self, x):
    # in lightning, forward defines the prediction/inference actions
    return self.probs_to_preds(model(x))

  def training_step(self, batch, batch_idx):
    # training_step defines the train loop. It is independent of forward
    x, y = batch
    loss, probs = self.model(x, y)
    preds = self.probs_to_preds(probs)
    self.train_metrics(preds, y)
    self.log('train_metrics', self.train_metrics)
    return loss
  
  def training_epoch_end(self, outputs):
    self.log('train_acc_epoch', self.train_metrics.compute())
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    y_hat = torch.argmax(y_hat, dim=3)
    #loss = F.cross_entropy(y_hat, y)
    #self.log('val_loss', loss)
  
  def validation_epoch_end(self, outputs):
    pass

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
    return optimizer


texts = data_loader.DataSet.load_json2(args.data).split_chunks().texts[:10]
tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
preprocessor = Preprocessor(tokenizer, BERT.max_length)
dataset = TextDataset(texts, preprocessor)

batch_size = 2
num_workers = 0
split_index = (len(dataset)*2)//3
train_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, sampler=SubsetRandomSampler(range(0, split_index)))
eval_loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, sampler=SubsetRandomSampler(range(split_index, len(dataset))))

lightning_model = LightningModel()
trainer = pl.Trainer()
trainer.fit(lightning_model, train_loader, eval_loader)