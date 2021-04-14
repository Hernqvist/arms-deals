import torch

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
        return y

    def labels(self, text, label="Buyer"):
        tokenized = self.tokenizer(text.text, return_offsets_mapping=True, **self.common_options())
        offsets = [(int(start), int(end)) for start, end in tokenized['offset_mapping'][0]]
        x = tokenized['input_ids'][0]
        y = self.token_has_label(text, label, offsets)
        if False and text.positive_sample:
            decoded = [self.tokenizer.decode(z) for z in x]
            print(self.tokenizer.decode(x))
            for a, b in zip(decoded, y):
                if b == 1:
                    print(a)
        return x, y
