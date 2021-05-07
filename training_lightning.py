import data_loader
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from preprocessor import Preprocessor
from transformers import BertTokenizerFast
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import torch.nn.functional as F
from parallel_token_classification import BertForParallelTokenClassification, LABELS

parser = argparse.ArgumentParser(description='Train a network to identify arms deals.')
parser.add_argument('data', type=str, help="The dataset directory.")
parser.add_argument('-l', '--load', default=None, type=str, help="Load the model from a file.")
parser.add_argument('-s', '--save', action='store_true', help="Save the model between epochs.")
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

class LitModule(pl.LightningModule):

  def __init__(self, config):
    super().__init__()
    self.model = BERT()
    self.save_hyperparameters(config)
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
    texts = data_loader.DataSet.load_json2(args.data).split_chunks().texts[:10]
    tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
    preprocessor = Preprocessor(tokenizer, BERT.max_length)
    self.dataset = TextDataset(texts, preprocessor)

    data_split_weights = [3, 1, 1]
    eval_start = int(len(self.dataset)*sum(data_split_weights[:1])/sum(data_split_weights))
    test_start = int(len(self.dataset)*sum(data_split_weights[:2])/sum(data_split_weights))
    self.train_sampler = SubsetRandomSampler(range(0, eval_start))
    self.val_sampler = SubsetRandomSampler(range(eval_start, test_start))
    self.test_sampler = SubsetRandomSampler(range(test_start, len(self.dataset)))

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
    return torch.argmax(probabilities, dim=3)
  
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

num_workers = 0
kwargs = {}
callbacks = []

if args.load:
  lit_module = LitModule.load_from_checkpoint(args.load)
  kwargs['resume_from_checkpoint'] = args.load
else:
  lit_module = LitModule({'lr':5e-5, 'batch_size':2})

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
    max_epochs=2,
    **kwargs)


trainer.fit(lit_module)
trainer.test(lit_module)