from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Extracts events from a tensorboard file')
parser.add_argument('file', type=str, help="The event file to load.")
parser.add_argument('events', metavar='N', type=str, nargs='+', help='The events to save')
args = parser.parse_args()


data = defaultdict(dict)
for entry in summary_iterator(args.file):
  value = entry.summary.value
  for i in value:
    if not i.tag in args.events:
      continue
    data[entry.step][i.tag] = i.simple_value
    data[entry.step]['step'] = entry.step

steps = [x for x in data]
steps.sort()
labels = ['step'] + args.events
print("\t".join(labels))
for step in steps:
  print("\t".join([str(data[step][label]).replace('.',',') for label in labels]))