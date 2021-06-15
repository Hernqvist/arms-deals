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
`python3 training_lightning.py --gpu --classifier albert --task sequence --max_epochs 100 --train_portion 0.95 --lr 0.00001 --batch_size 4 --max_tokens 128 --split fixed --test data.json`

Result:
`asdf`