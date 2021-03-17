import json

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
    'Answer':Colors.LIGHT_WHITE}

  def __init__(self, text, label_):
    self.start = label_['start']
    self.end = label_['end']
    self.content = text[self.start:self.end]
    self.type = label_['marker']

class Deal:
  def __init__(self, text, answer_, labels_):
    self.answer = Label(text, answer_)
    self.start, self.end = self.answer.start, self.answer.end
    self.buyers, self.sellers, self.prices, self.weapons = [], [], [], []
    start, end = self.answer.start, self.answer.end

    for label_ in labels_:
      if label_['start'] < start or label_['end'] > end or label_['marker'] == 'Answer':
        continue

      label = Label(text, label_)
      if label.type == "Buyer":
        self.buyers.append(label)
      elif label.type == "Seller":
        self.sellers.append(label)
      elif label.type == "Price":
        self.prices.append(label)
      elif label.type == "Weapon":
        self.weapons.append(label)

    self.all_labels = self.buyers + self.sellers + self.prices + self.weapons

class Text:
  def __init__(self, data_):
    self.deals = []
    self.text = data_['context']
    labels_ = next(iter(data_['labels'].values()))

    for label_ in labels_:
      if label_['marker'] == 'Answer':
        if label_['end'] - label_['start'] < 3:
          continue # It's just a placeholder
        self.deals.append(Deal(self.text, label_, labels_))
 
    self.positive_sample = len(self.deals) > 0

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
  def __init__(self, path):
    self.texts = []

    with open(path) as file:
      data = json.load(file)
      for data_ in data['data']:
        text = Text(data_)
        self.texts.append(text)


if __name__ == "__main__":
  dataset = DataSet("export.json")
  for text in dataset.texts:
    print("POSITIVE" if text.positive_sample else "NEGATIVE", "SAMPLE")
    print(text.colored_text()) 
    print()