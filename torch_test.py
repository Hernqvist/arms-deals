import data_loader
import sys
import torch
import torch.optim as optim
import sklearn.metrics
from preprocessor import Preprocessor
from transformers import BertTokenizerFast
from tabulate import tabulate

import torch.nn as nn
from transformers import BertForSequenceClassification, BertForTokenClassification

class BERT(nn.Module):
  max_length = 128
  options_name = "bert-base-cased"
  def __init__(self):
    super(BERT, self).__init__()
    self.encoder = BertForTokenClassification.from_pretrained(self.options_name)

  def forward(self, text, label):
    loss, text_fea = self.encoder(text, labels=label)[:2]
    return loss, text_fea

def preprocess_dataset(dataset):

  texts, labels = [], []

  for text in dataset.texts:
    #if not text.positive_sample:
    #  continue
    x, y = preprocessor.labels(text)
    texts.append(x)
    labels.append(y)
    #if text.positive_sample:
    #  print(preprocessor.print_labels(x, y))

  return torch.stack(texts), torch.stack(labels)


dataset = data_loader.DataSet.load_json2(sys.argv[1]).split_chunks()
tokenizer = BertTokenizerFast.from_pretrained(BERT.options_name)
preprocessor = Preprocessor(tokenizer, BERT.max_length)
texts, labels = preprocess_dataset(dataset)

split_at = (len(texts)*2)//3
eval_texts, eval_labels = texts[split_at:], labels[split_at:]
train_texts, train_labels = texts[:split_at], labels[:split_at]

print(len(train_texts), "Train")
print(len(eval_texts), "Eval")

model = BERT()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
BATCH_SIZE = 20
EPOCHS = 30
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
      predictions = torch.argmax(probabilities, dim=2)
      for x, y_actual, y in zip(eval_texts, eval_labels, predictions):
        preprocessor.print_labels(x, y_actual, y)
    if True:
      #predictions = torch.argmax(probabilities, dim=1).tolist()
      #actual = [x[0] for x in eval_labels.tolist()]
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

