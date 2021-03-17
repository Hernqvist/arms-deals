import argparse
import textract
import glob
import os
from pathlib import Path
from PyPDF2 import PdfFileReader
import progressbar
from newspaper import Article
import re
import langdetect

def scrape(text):
  regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
  urls = re.findall(regex, text)
  if len(urls) == 0:
    raise Exception("Found no urls")
  url = urls[-1]
  article = Article(url)
  article.download()
  article.parse()
  text = article.text
  if len(text) == 0:
    raise Exception("Got no text from", url)
  return text


parser = argparse.ArgumentParser(description='Make pdf data into text.')
parser.add_argument('dir', type=str, help="files to be processed")
parser.add_argument('-o', '--output', type=str, help="text output directory")
parser.add_argument('-l', '--limit', type=int, default=-1, help="max files to process")
parser.add_argument('--scrape', action='store_true', help="get text from url instead of pdf")
parser.add_argument('--verbose', action='store_true', help="print debug output")
parser.add_argument('--language', type=str, default=None, help="language code to filter")
args = parser.parse_args()

if args.output:
  Path(args.output).mkdir(parents=True, exist_ok=True)

processed = 0
paths = glob.glob(args.dir)[:args.limit]
for path in progressbar.progressbar(paths):
  try:
    with open(path, 'rb') as f:
      pdf = PdfFileReader(f)
      info = pdf.getDocumentInfo()
      meta = info['/Keywords'] if '/Keywords' in info else ""
      if not isinstance(meta, str):
        meta = meta.decode("utf-8")
      meta_params = {}
      for entry in meta.split():
        if entry.count(".") == 1:
          param, value = entry.split(".")
          meta_params[param] = value

    text = textract.process(path).decode("utf-8")
    if args.scrape:
      text = scrape(text)

    if args.language != None:
      language = langdetect.detect(text)
      if language != args.language:
        raise Exception("Wrong language: {} instead of {}".format(language, args.language))

    if args.output:
      filename = Path(path).stem

      for param in ["A", "B"]:
        if param in meta_params:
          filename += "_{}_{}".format(param, meta_params[param])

      outpath = os.path.join(args.output, filename + ".txt")
      with open(outpath, 'w+') as file:
        file.write(text)
    else:
      print()
      print("--->", path)
      print(text)
    processed += 1
  except Exception as e:
    if args.verbose:
      print("\nFAILED for", path, "with exception", e)
print(processed, "/", len(paths), "processed successfully.")