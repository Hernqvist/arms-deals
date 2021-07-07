# Arms deals classification

This is Fredrik Hernqvist's thesis project.

## Annotations
I'm annotating about 800 texts. The unprocessed text files are stored in `raw_data`. As I annotate texts, these get a matching json file in `annotated_data`. The json files are all compiled into `data.json`, which is what is used for training.

To start the editor, simply run `python3 editor.py`. Things might have the wrong scales depending on your resolution. `data.json` is automatically updated upon saving.

## Training
Training is done using `training_lightning.py`. It has a bunch of arguments which are mostly self explanatory, they can be viewed by running `python3 training_lightning.py --help`.

To run a basic example, run
```
python3 training_lightning.py data.json --max_epochs 5 --small_data --print
```

This will train the BERT model for 5 epochs and print the classification of the trained model. The `--small_data` argument crops the entire dataset to 20 texts. To view the metrics (this can be done live during training), run  ```tensorboard --logdir lightning/lightning_logs/``` and open [http://localhost:6006/](http://localhost:6006/) in the browser.

We can also change the arguments to use ALBERT instead of BERT using the `--classifier` argument.

### Sequence classification
When setting `--task sequence` we are trying to predict wether the entire text contains an arms deal or not. It outputs either 0 or 1 for each text.

### Token classification
When setting `--task token` we are trying to predict whether each token is a certain attribute of the arms deal. For each text, it outputs an `n*8` matrix. The value at index `(i, j)` in this matrix is an 1 if token `i` is an instance of attribute `j`, otherwise 0. The attributes are Weapon, Buyer, Buyer Country, Seller, Seller Country, Quantity, Price, Date, in that order.


## Text Extractor
`text_extractor.py` was used to get the raw text from the pdf files. It looks at the pdf file, finds the url at the bottom, and uses the `newspaper` module to download the text from its source. This is not needed anymore.

## Tests
Here are the commands for running the tests in the report and the results.
### Sequence classification with exactly 400 character chunks of text
Command:
`python3 training_lightning.py --gpu --classifier albert --task sequence --max_epochs 100 --train_portion 0.8 --lr 0.00001 --batch_size 4 --max_tokens 128 --split fixed --test data.json`

Result:
```
{'TestAccuracy': 0.7857142686843872,
 'TestF1': 0.8167938590049744,
 'TestLoss': 0.5804754495620728,
 'TestPrecision': 0.8045112490653992,
 'TestRecall': 0.8294573426246643}
```

### Token classification with exactly 400 character chunks of text
Command:
`python3 training_lightning.py --gpu --classifier albert --task token --max_epochs 200 --train_portion 0.8 --lr 0.00001 --batch_size 8 --max_tokens 128 --split fixed --test data.json`

Result:
```
{'TestAccuracy': 0.9911141395568848,
 'TestF1': 0.6636696457862854,
 'TestLoss': 0.06833155453205109,
 'TestPrecision': 0.7458832263946533,
 'TestRecall': 0.5977804660797119}
```

### Sequence classification with variable chunks of text
Command:
`python3 training_lightning.py --gpu --classifier albert --task sequence --max_epochs 100 --train_portion 0.8 --lr 0.00001 --batch_size 4 --max_tokens 128 --split chunks --test data.json`

Result:
```
{'TestAccuracy': 0.8706896305084229,
 'TestF1': 0.8110831379890442,
 'TestLoss': 0.44118180871009827,
 'TestPrecision': 0.8429319262504578,
 'TestRecall': 0.7815533876419067}
```

### Token classification with variable chunks of text
Command:
`python3 training_lightning.py --gpu --classifier albert --task token --max_epochs 200 --train_portion 0.8 --lr 0.00001 --batch_size 8 --max_tokens 128 --split chunks --test data.json`

Result:
```
{'TestAccuracy': 0.9967487454414368,
 'TestF1': 0.7586552500724792,
 'TestLoss': 0.01827111281454563,
 'TestPrecision': 0.8267502188682556,
 'TestRecall': 0.7009238004684448}
```
 