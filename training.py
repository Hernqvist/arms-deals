import data_loader
import sys
import torch
import torch.optim as optim
import sklearn.metrics
from torch.utils.data import DataLoader, Dataset
from preprocessor import Preprocessor
from transformers import BertTokenizerFast
from tabulate import tabulate
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, Precision, Recall

import torch.nn as nn
from parallel_token_classification import BertForParallelTokenClassification

LABELS = ("Buyer", "Seller", "Weapon", "Price")

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


model = BERT()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
texts = data_loader.DataSet.load_json2(sys.argv[1]).split_chunks().texts[:4]
tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
preprocessor = Preprocessor(tokenizer, BERT.max_length)
dataset = TextDataset(texts, preprocessor)
train_loader = DataLoader(dataset, batch_size=2)
eval_loader = DataLoader(dataset, batch_size=2)

def train_step(engine, batch):
  model.train()
  optimizer.zero_grad()
  x, y = batch
  loss, _ = model(x, y)
  loss.backward()
  optimizer.step()
  return loss.item()
trainer = Engine(train_step)

def validation_step(engine, batch):
  model.eval()
  with torch.no_grad():
    x, y = batch
    probabilities = model(x, y)[1]
    y_pred = torch.argmax(probabilities, dim=3)
    return y_pred, y
evaluator = Engine(validation_step)
Accuracy().attach(evaluator, "accuracy")
precision = Precision()
recall = Recall()
#F1 = (precision * recall * 2 / (precision + recall)).mean()
#F1.attach(evaluator, "F1")
precision.attach(evaluator, "precision")
recall.attach(evaluator, "recall")

@trainer.on(Events.ITERATION_COMPLETED(every=1))
def log_training_loss(trainer):
  print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.4f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
  evaluator.run(train_loader)
  metrics = evaluator.state.metrics
  print(f"Training Results - Epoch: {trainer.state.epoch}  Accuracy: {metrics['accuracy']:.4f} F1: {metrics['accuracy']:.4f} precision: {metrics['precision']:.4f} recall: {metrics['recall']:.4f}")

trainer.run(train_loader, max_epochs=100)


print(len(train_texts), "Train")
print(len(eval_texts), "Eval")

BATCH_SIZE = 20
EPOCHS = 300
eval_every = 100
until_eval, total_evaled, total = eval_every, 0, len(train_texts)*EPOCHS

def scores(texts, labels):
  output = model(texts, labels)
  loss, probabilities = output
  loss = loss.item()
  if len(probabilities.size()) == 3:
    predictions = torch.argmax(probabilities, dim=2)
  else:
    predictions = torch.argmax(probabilities, dim=1)
  predicted = predictions.flatten()
  actual = labels.flatten()
  precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(actual, predicted)
  return loss, precision[1], recall[1], f1[1]

history = []
def eval():
  global total_evaled, until_eval, eval_every
  print("Processed", total_evaled, "/", total, "datapoints. Evaluating.")
  until_eval = eval_every
  model.eval()
  with torch.no_grad():
    if True:   
      output = model(eval_texts, eval_labels)
      loss, probabilities = output
      predictions = torch.argmax(probabilities, dim=3)
      print("In eval")
      print(probabilities.size())
      print(predictions.size())
      print(eval_labels.size())
      print(eval_texts.size())
      
      for x, y_actual_all, y_all in zip(eval_texts, eval_labels, predictions):
        print()
        for label, y_actual, y in zip(labels_considered, torch.transpose(y_actual_all, 0, 1), torch.transpose(y_all, 0, 1)):
          print((label.upper() + " ")*5)
          preprocessor.print_labels(x, y_actual, y)
    if False:
      print("Calculating scores")
      e_loss, e_precision, e_recall, e_f1 = scores(eval_texts, eval_labels)
      t_loss, t_precision, t_recall, t_f1 = scores(train_texts, train_labels)
      print(tabulate([
        ["Training", t_loss, t_precision, t_recall, t_f1],
        ["Evaluation", e_loss, e_precision, e_recall, e_f1]
      ], headers=[' ', 'Loss', 'Precision', 'Recall', 'F1'], tablefmt='orgtbl'))
      print()
      history.append([total_evaled, e_f1, t_f1])
      print(tabulate(history, headers=['Points', 'Eval F1', 'Train F1'], tablefmt='orgtbl'))

eval()
for epoch in range(EPOCHS):
  print("Epoch", epoch)
  for batch in range(len(train_texts)//BATCH_SIZE):
    start, stop = BATCH_SIZE * batch, BATCH_SIZE * (batch+1)
    batch_texts = train_texts[start:stop]
    batch_labels = train_labels[start:stop]

    model.train()
    output = model(batch_texts, batch_labels)
    loss, _ = output

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    until_eval -= len(batch_texts)
    total_evaled += len(batch_texts)
    if until_eval <= 0:
      eval()
print("Done.")
eval()

