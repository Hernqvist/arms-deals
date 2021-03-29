import data_loader
import sys
import torch
import torch.optim as optim
from transformers import BertTokenizer
from bert_module import BERT

def preprocess_dataset(dataset):
  tokenizer = BertTokenizer.from_pretrained(BERT.options_name)
  PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

  texts, labels = [], []

  for text in dataset.texts:
    encoded = torch.LongTensor(tokenizer(text.text, max_length=BERT.max_length, truncation=True, padding='max_length')['input_ids'])
    positive = torch.LongTensor([1 if text.positive_sample else 0])
    texts.append(encoded)
    labels.append(positive)

  return torch.stack(texts), torch.stack(labels)


dataset = data_loader.DataSet.load_json(sys.argv[1])
texts, labels = preprocess_dataset(dataset)

split_at = len(texts)//3
eval_texts, eval_labels = texts[:split_at], labels[:split_at]
train_texts, train_labels = texts[split_at:], labels[split_at:]

model = BERT()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
BATCH_SIZE = 5
EPOCHS = 10
eval_every = 20
until_eval, total_evaled = eval_every, 0

def eval():
  global total_evaled, until_eval, eval_every
  print("Processed", total_evaled, "datapoints. Evaluating.")
  until_eval = eval_every
  model.eval()
  with torch.no_grad():                    
    output = model(eval_texts, eval_labels)
    loss, _ = output
    print("Loss: ", loss.item())


print(len(texts))
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
      

