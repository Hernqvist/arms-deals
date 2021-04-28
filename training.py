import data_loader
import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from preprocessor import Preprocessor
from transformers import BertTokenizerFast
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.contrib.handlers import ProgressBar
import torch.nn as nn
from parallel_token_classification import BertForParallelTokenClassification, LABELS

parser = argparse.ArgumentParser(description='Train a network to identify arms deals.')
parser.add_argument('data', type=str, help="The dataset directory.")
args = parser.parse_args()

class BERT(nn.Module):
  max_length = 128
  options_name = "bert-base-cased"

  def __init__(self):
    super(BERT, self).__init__()
    self.encoder = BertForParallelTokenClassification.from_pretrained(self.options_name)

  def forward(self, text, label):
    loss, text_fea = self.encoder(text, labels=label)[:2]
    return loss, text_fea

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

def avoid_zero_predicts(y_pred, y):
  if y_pred.sum() == 0:
    y_pred[0][0][0] = 1
    y[0][0][0] = 0

model = BERT()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
texts = data_loader.DataSet.load_json2(args.data).split_chunks().texts[:16]
tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
preprocessor = Preprocessor(tokenizer, BERT.max_length)
dataset = TextDataset(texts, preprocessor)

split_index = (len(dataset)*2)//3
train_loader = DataLoader(dataset, batch_size=2, sampler=SubsetRandomSampler(range(0, split_index)))
eval_loader = DataLoader(dataset, batch_size=2, sampler=SubsetRandomSampler(range(split_index, len(dataset))))

def train_step(engine, batch):
  model.train()
  optimizer.zero_grad()
  x, y = batch
  loss, _ = model(x, y)
  loss.backward()
  optimizer.step()
  return loss.item()
trainer = Engine(train_step)
ProgressBar().attach(trainer)

def validation_step(engine, batch):
  model.eval()
  with torch.no_grad():
    x, y = batch
    probabilities = model(x, y)[1]
    y_pred = torch.argmax(probabilities, dim=3)
    avoid_zero_predicts(y_pred, y)
    return y_pred, y
evaluator = Engine(validation_step)
ProgressBar().attach(evaluator)
Accuracy().attach(evaluator, "accuracy")
precision = Precision()
recall = Recall()
F1 = (precision * recall * 2 / (precision + recall)).mean()
F1.attach(evaluator, "F1")
precision.attach(evaluator, "precision")
recall.attach(evaluator, "recall")

def print_step(engine, batch):
  model.eval()
  with torch.no_grad():
    x, y = batch
    probabilities = model(x, y)[1]
    y_pred = torch.argmax(probabilities, dim=3)

    
    for x_text, y_text, y_pred_text in zip(x, y, y_pred):
      print()
      for label, y_label, y_pred_label in zip(LABELS, 
          torch.transpose(y_text, 0, 1),
          torch.transpose(y_pred_text, 0, 1)):
        print("\033[7m", label, "\033[0m")
        preprocessor.print_labels(x_text, y_label, y_pred_label)
    print()
printer = Engine(print_step)

def print_metrics(trainer, metrics, headline):
  print((
      f"{headline} Results - Epoch: {trainer.state.epoch}\t"
      f" Accuracy: {metrics['accuracy']:.4f}"
      f" F1: {metrics['F1']:.4f}"
      f" Precision: {metrics['precision']:.4f}"
      f" Recall: {metrics['recall']:.4f}"))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
  printer.run(eval_loader)
  evaluator.run(train_loader)
  print_metrics(trainer, evaluator.state.metrics, "Training")
  evaluator.run(eval_loader)
  print_metrics(trainer, evaluator.state.metrics, "Validation")

@trainer.on(Events.STARTED)
def on_start(trainer):
  log_training_results(trainer)

trainer.run(train_loader, max_epochs=100)

print(len(train_texts), "Train")
print(len(eval_texts), "Eval")

