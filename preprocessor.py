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
