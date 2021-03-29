import torch.nn as nn
from transformers import BertForSequenceClassification

class BERT(nn.Module):

  max_length = 128
  options_name = "bert-base-uncased"

  def __init__(self):
    super(BERT, self).__init__()

    self.encoder = BertForSequenceClassification.from_pretrained(self.options_name)

  def forward(self, text, label):
    loss, text_fea = self.encoder(text, labels=label)[:2]

    return loss, text_fea