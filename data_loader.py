import json
import sys
import functools
import random

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32

class LabelType:
  def __init__(self, tag, color, priority):
    self.tag = tag
    self.color = color
    self.priority = priority

label_types = {}
def add_layer_type(tag, color):
  label_types[tag] = LabelType(tag, color, len(label_types))

add_layer_type("Buyer Country", Colors.LIGHT_GREEN)
add_layer_type("Buyer", Colors.YELLOW)
add_layer_type("Seller Country", Colors.LIGHT_BLUE)
add_layer_type("Seller", Colors.BLUE)
add_layer_type("Price", Colors.CYAN)
add_layer_type("Quantity", Colors.LIGHT_PURPLE)
add_layer_type("Weapon", Colors.RED)
add_layer_type("Date", Colors.DARK_GRAY)
add_layer_type("Answer", Colors.END)

ignore_labels = {"Buyer Country", "Seller Country"}

# Variables that correspond directly to a json element are suffixed with _

class Label:
  trimmed_chars = " ,.-"
  def __init__(self, text, start, end, type):
    while text[start] in self.trimmed_chars:
      start += 1
    while text[end-1] in self.trimmed_chars:
      end -= 1
    self.content = text[start:end]
    self.type = type
    self.start = start
    self.end = end

  @classmethod
  def from_json(cls, text, label_):
    start, end = label_['start'], label_['end']
    return cls(text, start, end, label_['marker'])
  
  @classmethod
  def from_json2(cls, text, label_):
    start, end = label_['start'], label_['end']
    return cls(text, start, end, label_['tag'])

  @classmethod
  def from_transpose(cls, label, text, offset):
    return cls(text[offset:], label.start - offset, label.end - offset, label.type)

class Deal:
  def __init__(self, answer, labels):
    self.answer = answer
    self.start, self.end = self.answer.start, self.answer.end
    self.labels = {tag:[] for tag in label_types}

    for label in labels:
      if label.start < self.start or label.end > self.end \
          or label.type == 'Answer' or label.type in ignore_labels:
        continue

      self.labels[label.type].append(label) 

    self.all_labels = functools.reduce(lambda x,y: x+y, self.labels.values())

  @classmethod
  def from_json(cls, text, answer_, labels_):
    answer = Label.from_json(text, answer_)
    labels = [Label.from_json(text, label_) for label_ in labels_]
    return cls(answer, labels)
  
  @classmethod
  def from_json2(cls, text, answer_, labels_):
    answer = Label.from_json2(text, answer_)
    labels = [Label.from_json2(text, label_) for label_ in labels_]
    return cls(answer, labels)

  @classmethod
  def from_transpose(cls, deal, text, offset):
    labels = [Label.from_transpose(label, text, offset) for label in deal.all_labels]
    return cls(Label.from_transpose(deal.answer, text, offset), labels)

class Text:
  def __init__(self, text, deals):
    self.text = text
    self.deals = sorted(deals, key=lambda deal: deal.start)
    self.positive_sample = len(self.deals) > 0

  @classmethod
  def from_json(cls, data_):
    text = data_['context']
    labels_ = next(iter(data_['labels'].values()))

    deals = []
    for label_ in labels_:
      if label_['marker'] == 'Answer':
        if label_['end'] - label_['start'] < 3:
          continue # It's just a placeholder
        deals.append(Deal.from_json(text, label_, labels_))

    return cls(text, deals)
  
  @classmethod
  def from_json2(cls, data_):
    text = data_['text']
    labels_ = data_['labels']

    deals = []
    for label_ in labels_:
      if label_['tag'] == 'Answer':
        deals.append(Deal.from_json2(text, label_, labels_))

    return cls(text, deals)

  def split_chunks(self, min_length):
    if len(self.deals) > 1:
      return [text for one_deal in self.split_deals() for text in one_deal.split_chunks(min_length)]
    
    if len(self.deals) == 1:
      deal = self.deals[0]
      if deal.start > len(self.text) - deal.end:
        index = deal.start//2
      else:
        index = (len(self.text) + deal.end)//2
    else:
      index = len(self.text)//2
    
    if index < min_length or len(self.text) - index < min_length:
      return [self]
    text1, text2 = self.split(index)
    return text1.split_chunks(min_length) + text2.split_chunks(min_length)

  # Splits the text between deals so that each text contains at most one deal
  def split_deals(self):
    if len(self.deals) < 2:
      return [self]
    head, tail = self.split((self.deals[0].end + self.deals[1].start)//2)
    return [head] + tail.split_deals()
  
  def split(self, index):
    deals1, deals2 = [], []
    text1, text2 = self.text[:index], self.text[index:]
    for deal in self.deals:
      if deal.start <= index and index < deal.end:
        raise Exception("Can't split a text through a deal")
      if deal.end <= index:
        deals1.append(Deal.from_transpose(deal, self.text, 0))
      else:
        deals2.append(Deal.from_transpose(deal, self.text, index))
    return (Text(text1, deals1), Text(text2, deals2))

      
  def colored_text(self):
    ANSWER_COLOR = Colors.END + Colors.UNDERLINE
    labels = [""]*len(self.text)

    for deal in self.deals:
      for i in range(deal.start, deal.end):
        labels[i] = "Answer"
      for label in deal.all_labels:
        for i in range(label.start, label.end):
          if labels[i] == "" or label_types[labels[i]].priority > label_types[label.type].priority:
            labels[i] = label.type

    text = ""
    prev_label = ""
    for label, char in zip(labels, self.text):
      modifier = Colors.END + label_types[label].color + Colors.UNDERLINE if label != "" else ""
      if label == "" and prev_label != "":
        modifier = Colors.END
      text = text + modifier + char
      prev_label = label
    text += Colors.END
    return text

class DataSet:
  def __init__(self, texts, name):
    self.texts = texts
    self.name = name

  @classmethod
  def load_json(cls, path):
    texts = []
    with open(path) as file:
      data = json.load(file)
      for data_ in data['data']:
        text = Text.from_json(data_)
        texts.append(text)
    return cls(texts, path)
  
  @classmethod
  def load_json2(cls, path):
    texts = []
    with open(path) as file:
      data = json.load(file)
      for data_ in data:
        text = Text.from_json2(data_)
        texts.append(text)
    return cls(texts, path)

  def split_deals(self):
    texts = []
    for text in self.texts:
      texts.extend(text.split_deals())
    return DataSet(texts, "Split of {}".format(self.name))
  
  def split_chunks(self, min_length=400, shuffle=True):
    texts = []
    for text in self.texts:
      texts.extend(text.split_chunks(min_length))
    if shuffle:
      random.Random(0).shuffle(texts)
    return DataSet(texts, "Split of {} into chunks of at least {}".format(self.name, min_length))

  def print_data(self):
    print("Dataset:", self.name)
    
    for text in self.texts:
      print(Colors.BOLD, Colors.UNDERLINE, "POSITIVE" if text.positive_sample else "NEGATIVE", " SAMPLE", Colors.END, sep="")
      print(text.colored_text()) 
      print()

    print(len([x for x in self.texts if x.positive_sample]), "positive samples")
    print(len([x for x in self.texts if not x.positive_sample]), "negative samples")

if __name__ == "__main__":
  dataset = DataSet.load_json2(sys.argv[1])
  split = dataset.split_chunks()
  split.print_data()