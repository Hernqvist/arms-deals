import json
import sys

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


# Variables that correspond directly to a json element are suffixed with _

class Label:
  colors = {
    'Buyer':Colors.YELLOW,
    'Seller':Colors.BLUE,
    'Price':Colors.CYAN,
    'Weapon':Colors.RED,
    'Answer':Colors.END}

  def __init__(self, content, start, end, type):
    self.start = start
    self.end = end
    self.content = content
    self.type = type

  @classmethod
  def from_json(cls, text, label_):
    start, end = label_['start'], label_['end']
    return cls(text[start:end], start, end, label_['marker'])

  @classmethod
  def from_transpose(cls, label, offset):
    return cls(label.content, label.start - offset, label.end - offset, label.type)

class Deal:
  def __init__(self, answer, labels):
    self.answer = answer
    self.start, self.end = self.answer.start, self.answer.end
    self.buyers, self.sellers, self.prices, self.weapons = [], [], [], []

    for label in labels:
      if label.start < self.start or label.end > self.end or label.type == 'Answer':
        continue

      if label.type == "Buyer":
        self.buyers.append(label)
      elif label.type == "Seller":
        self.sellers.append(label)
      elif label.type == "Price":
        self.prices.append(label)
      elif label.type == "Weapon":
        self.weapons.append(label)

    self.all_labels = self.buyers + self.sellers + self.prices + self.weapons

  @classmethod
  def from_json(cls, text, answer_, labels_):
    answer = Label.from_json(text, answer_)
    labels = [Label.from_json(text, label_) for label_ in labels_]
    return cls(answer, labels)

  @classmethod
  def from_transpose(cls, deal, offset):
    labels = [Label.from_transpose(label, offset) for label in deal.all_labels]
    return cls(Label.from_transpose(deal.answer, offset), labels)

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

  # Splits the text between deals so that each text contains at most one deal
  def split_deals(self):
    if not self.positive_sample:
      return [self]

    cutoffs = [0]
    for i in range(1, len(self.deals)):
      cutoffs.append((self.deals[i-1].end + self.deals[i].start)//2)
    cutoffs.append(len(self.text))

    texts = []
    for i in range(len(cutoffs) - 1):
      start, end = cutoffs[i], cutoffs[i+1]
      substring = self.text[start:end]
      deal = Deal.from_transpose(self.deals[i], start)
      texts.append(Text(substring, [deal]))

    return texts
      

  def colored_text(self):
    ANSWER_COLOR = Label.colors['Answer'] + Colors.UNDERLINE
    chars = [self.text[i] for i in range(len(self.text))]

    for deal in self.deals:
      for label in deal.all_labels:
        color = Label.colors[label.type]
        chars[label.start] = color + chars[label.start] + Colors.UNDERLINE
        chars[label.end - 1] = chars[label.end - 1] + ANSWER_COLOR
      chars[deal.start] = ANSWER_COLOR + chars[deal.start]
      chars[deal.end - 1] = chars[deal.end - 1] + Colors.END

    return "".join(chars)

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

  def split_deals(self):
    texts = []
    for text in self.texts:
      print(texts, "{", text.split_deals(), "}")
      texts.extend(text.split_deals())
    return DataSet(texts, "Split of {}".format(self.name))

  def print_data(self):
    print("Dataset:", self.name)
    
    for text in self.texts:
      print(Colors.BOLD, Colors.UNDERLINE, "POSITIVE" if text.positive_sample else "NEGATIVE", " SAMPLE", Colors.END, sep="")
      print(text.colored_text()) 
      print()

    print(len([x for x in self.texts if x.positive_sample]), "positive samples")
    print(len([x for x in self.texts if not x.positive_sample]), "negative samples")

if __name__ == "__main__":
  dataset = DataSet.load_json(sys.argv[1])
  split = dataset.split_deals()
  split.print_data()