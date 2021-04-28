import torch

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

class Preprocessor:

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def common_options(self):
        return {'max_length':self.max_length,
                'padding':'max_length',
                'truncation':True,
                'return_tensors':'pt'}
    
    # Text as imported from the data importer module
    def binary(self, text):
        x = self.tokenizer(text.text, **self.common_options())['input_ids'][0]
        y = torch.LongTensor([1 if text.positive_sample else 0])
        return x, y

    def has_overlap(self, x1, x2, y1, y2):
        return not (y1 >= x2 or y2 <= x1)

    def token_has_label(self, text, label, offsets):
        y = torch.zeros(len(offsets), dtype=torch.long)
        for i, (start, end) in enumerate(offsets):
            for deal in text.deals:
                for label_ in deal.labels[label]:
                    start_, end_ = label_.start, label_.end
                    if self.has_overlap(start, end, start_, end_):
                        y[i] = 1
        y[0] = 1 if text.positive_sample else 0
        return y

    def labels(self, text, label="Buyer"):
        tokenized = self.tokenizer(text.text, return_offsets_mapping=True, **self.common_options())
        offsets = [(int(start), int(end)) for start, end in tokenized['offset_mapping'][0]]
        x = tokenized['input_ids'][0]
        y = self.token_has_label(text, label, offsets)
        return x, y

    def labels_multiple(self, text, labels=("Buyer",)):
        tokenized = self.tokenizer(text.text, return_offsets_mapping=True, **self.common_options())
        offsets = [(int(start), int(end)) for start, end in tokenized['offset_mapping'][0]]
        x = tokenized['input_ids'][0]
        y = torch.stack([self.token_has_label(text, label, offsets) for label in labels])
        return x, torch.transpose(y, 0, 1)
    
    def print_labels(self, x, y_actual, y=None):
        if y == None:
            y = y_actual
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"
        NEGATIVE = "\033[7m"
        END = "\033[0m"
        decoded = [self.tokenizer.decode(z) for z in x]
        try:
            index = decoded.index("[PAD]")
            decoded = decoded[:index]
            y = y[:index]
            y_actual = y_actual[:index]
        except ValueError:
            pass

        output = []
        for token, predicted, actual in zip(decoded, y, y_actual):
            rep = token
            if actual == 1 and predicted == 1:
                rep = Colors.GREEN + rep + Colors.END
            if actual == 1 and predicted == 0:
                rep = Colors.YELLOW + rep + Colors.END
            if actual == 0 and predicted == 1:
                rep = Colors.RED + rep + Colors.END
            output.append(rep)
        print(" ".join(output))
