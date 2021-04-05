import os
import glob
import json
import data_loader
from pathlib import Path

INPUT_DIR = "raw_data"
OUTPUT_DIR = "annotated_data"
SALVAGE = "export.json"
cutoff = 100

def output_path(filename):
  return OUTPUT_DIR + "/" + filename + ".json"

def output_text(filename, data, content):
  global salvaged
  count = []
  for line in content.splitlines():
    if line == "":
      count.extend([0])
    else:
      count.extend([1]*(len(line) + 1))
  def real_index(index):
    i = 0
    while index > 0 or count[i] == 0:
      index -= count[i]
      i += 1
    return i
  salvaged += 1
  print("Salvaging", filename)

  as_json = {'filename': filename,
      'text': content}
  labels = []
  for label in data.all_labels:
    start = real_index(label.start)
    end = real_index(label.end)
    labels.append({'start':start,
      'end':end,
      'tag':label.type,
      'text':content[start:end]})
  as_json['labels'] = labels

  path = output_path(filename)
  with open(path, 'w') as file:
    json.dump(as_json, file, indent=2, sort_keys=True)

file_by_text = {}
unedited_by_text = {}
paths = glob.glob(INPUT_DIR + "/*")
for path in paths:
  with open(path) as file:
    content = file.read()
    edited = os.linesep.join([s for s in content.splitlines() if s])
    file_by_text[edited] = Path(path).stem
    unedited_by_text[edited] = content

print(len(file_by_text))

dataset = data_loader.DataSet.load_json(SALVAGE)
salvaged = 0
for text in dataset.texts:
  sample = text.text
  if sample in file_by_text:
    output_text(file_by_text[sample], text, unedited_by_text[sample])



print("salvaged", salvaged, "texts.")