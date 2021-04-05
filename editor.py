import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.font import Font
from tkinter.messagebox import askquestion
from collections import defaultdict
import os
import glob
import random
import json
from pathlib import Path

INPUT_DIR = "raw_data"
OUTPUT_DIR = "annotated_data"

class MultiSet:
  def __init__(self):
    self.count = defaultdict(lambda: 0)

  def as_list(self):
    l = []
    for x in self.count:
      l.extend([x]*max(0, self.count[x]))
    return l

  def insert(self, x):
    self.count[x] += 1

  def remove(self, x):
    self.count[x] -= 1

class Label:
  def __init__(self, start, end, tag):
    self.start = start
    self.end = end
    self.tag = tag

class LabelType:
  def __init__(self, tag, short_tag, key, color):
    self.tag = tag
    self.short_tag = short_tag
    self.key = key
    self.color = color

def blend_colors(colors):
  r, g, b = 0, 0, 0
  for color in colors:
    c_r, c_g, c_b = root.winfo_rgb(color)
    r += c_r
    g += c_g
    b += c_b
  f = 256 * len(colors)
  return "#{:02X}{:02X}{:02X}".format(r//f, g//f, b//f)

label_by_tag = {}
label_by_key = {}
def add_label_type(label_type):
  label_by_tag[label_type.tag] = label_type
  label_by_key[label_type.key] = label_type

add_label_type(LabelType("Answer", "A", 'a', 'grey80'))
add_label_type(LabelType("Weapon", "W", 'w', 'red'))
add_label_type(LabelType("Buyer", "B", 'b', 'yellow'))
add_label_type(LabelType("Buyer Country", "BC", 'y', 'brown'))
add_label_type(LabelType("Seller", "S", 's', 'blue'))
add_label_type(LabelType("Seller Country", "SC", 'x', 'cyan'))
add_label_type(LabelType("Quantity", "Q", 'q', 'magenta'))
add_label_type(LabelType("Price", "P", 'c', 'green'))
add_label_type(LabelType("Date", "D", 'd', 'purple'))

class BeginMark(tk.Canvas):
  def __init__(self, master, label):
    label_type = label_by_tag[label.tag]
    width, height = int(8*root.scale), int(16*root.scale)
    super().__init__(master=master,
        width=width, 
        height=height, 
        background='white',
        highlightthickness=0)
    self.create_oval(width-height/2, 0, width+height/2, height, 
        fill=blend_colors(['black', label_type.color]),
        outline='')
    self.create_text(width/2, height/2,
       fill='white',
       font="Arial {}".format(int(height*0.3)),
       text=label_type.short_tag,
       angle=90)

class EndMark(tk.Canvas):
  def __init__(self, master, label, callback):
    label_type = label_by_tag[label.tag]
    width, height = int(8*root.scale), int(16*root.scale)
    super().__init__(master=master,
        width=width, 
        height=height, 
        background='white',
        highlightthickness=0)
    self.create_oval(-height/2, 0, height/2, height, 
        fill=blend_colors(['black', label_type.color]),
        outline='')
    self.create_text(width/2, height/2,
       fill='black',
       font="Arial {}".format(int(height*0.3)), text="X")
    self.bind("<Button-1>", callback)

class Editor(tk.Text):
  def __init__(self, master=None):
    super().__init__(master=master, wrap="word")
    self.master = master
    self.on_edit = lambda: None

    self.initialize("")

  def refresh(self):
    y = self.yview()[0]

    self.windows = []
    self.config(state='normal')
    self.configure(font=Font(family="Times New Roman",
        size=int(10*root.scale)))
    for tag in self.tag_names():
      self.tag_delete(tag)

    for window in self.windows:
      window.destroy()

    self.delete('1.0', 'end')
    self.insert('1.0', self.text)

    text_colors = [(len(self.text), '-end')]

    for i, label in enumerate(self.labels):
      tag_name = 'label_{}'.format(i)
      self.mark_set(tag_name + '_start', self.n_chars(label.start))
      self.mark_set(tag_name + '_end', self.n_chars(label.end))
      label_type = label_by_tag[label.tag]
      text_colors.append((label.start, label_type.color))
      text_colors.append((label.end, '-' + label_type.color))

    text_colors.sort()
    color_mix = MultiSet()
    last_i = 0
    next_id = 0
    for i, color_event in text_colors:
      colors = color_mix.as_list()
      if len(colors) > 0:
        start, end = self.n_chars(last_i), self.n_chars(i)
        tag_name = 'tag_{}'.format(next_id)
        next_id += 1
        self.tag_add(tag_name, start, end)
        self.tag_configure(tag_name,
            background=blend_colors(['white'] + colors))
      last_i = i
      if color_event[0] == '-':
        color_mix.remove(color_event[1:])
      else:
        color_mix.insert(color_event)

    for i, label in enumerate(self.labels):
      tag_name = 'label_{}'.format(i)
      begin = BeginMark(self, label)
      end = EndMark(self, label, lambda _, i=i: self.delete_label(i))
      self.windows.extend([begin, end])
      self.window_create(tag_name + '_start', window=begin)
      self.window_create(tag_name + '_end', window=end)

    self.config(state='disabled')
    self.yview_moveto(y)

  def add_label(self, label):
    self.labels.append(label)
    self.refresh()
    self.on_edit()

  def delete_label(self, i):
    self.labels.pop(i)
    self.refresh()
    self.on_edit()

  def initialize(self, text):
    self.text = text
    self.labels = []
    self.refresh()

  def key_pressed(self, event):
    char = event.char
    if not char in label_by_key:
      return
    label_type = label_by_key[char]
    ranges = self.tag_ranges('sel')
    if len(ranges) == 0:
      return
    start, end = map(self.to_integer_index, ranges)
    self.add_label(Label(start, end, label_type.tag))

  def to_integer_index(self, text_index):
    index = 0
    row, col = map(int, str(text_index).split('.'))
    for i, line in enumerate(self.text.split('\n')):
      i += 1
      if i < row:
        index += len(line) + 1
      else:
        index += col
        for window in self.window_names():
          wrow, wcol = map(int, str(self.index(window)).split('.'))
          if wrow == row and wcol <= col:
            index -= 1
        break
    return index

  def n_chars(self, n):
    return '1.0 + {} chars'.format(n)

  def get_json(self):
    labels = []
    for label in self.labels:
      json_label = {}
      json_label['start'] = label.start
      json_label['end'] = label.end
      json_label['tag'] = label.tag
      json_label['text'] = self.text[label.start:label.end]
      labels.append(json_label)
    return {'text':self.text, 'labels':labels}

  def load_json(self, data):
    for json_label in data['labels']:
      self.labels.append(Label(
          json_label['start'],
          json_label['end'],
          json_label['tag']))
    self.refresh()

class LegendTag(tk.Label):
  def __init__(self, master, tag):
    label_type = label_by_tag[tag]
    super().__init__(master=master,
        text="{} ({})".format(tag, label_type.key.upper()),
        fg='black',
        bg=blend_colors([label_type.color, 'gray90']))

class Legend(tk.Frame):
  def __init__(self, master):
    super().__init__(master=master)
    self.save_button = tk.Button(self, text="Save")
    self.save_button.pack(side='left', padx=5, pady=5)
    self.next_button = tk.Button(self, text="New text")
    self.next_button.pack(side='left', padx=5, pady=5)

    for tag in label_by_tag:
      LegendTag(self, tag).pack(side='left', padx=5)

class Application(tk.Frame):
  def key_pressed(self, event):
    self.editor.key_pressed(event)

  def __init__(self, master=None):
    super().__init__(master)
    self.master = master
    self.winfo_toplevel().title("Textinatiratiror") 
    self.pack(expand=True, fill='both')

    self.legend = Legend(self)
    self.legend.pack(side='bottom', fill='x')
    self.legend.save_button.configure(command=self.save)
    self.legend.next_button.configure(command=self.load_random)

    self.editor = Editor(self)
    self.editor.pack(expand=True, fill='both',
        padx=5, pady=5)
    self.editor.on_edit = self.invalidate
    self.master.bind("<Key>", self.editor.key_pressed)

    if not os.path.exists(INPUT_DIR):
      os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)

    menubar = tk.Menu(master)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=self.load_specific)
    filemenu.add_command(label="Delete", command=self.delete_current)
    menubar.add_cascade(label="File", menu=filemenu)
    root.config(menu=menubar)

    self.load_random()

  def save(self):
    self.legend.next_button["state"] = "normal"
    as_json = self.editor.get_json()
    as_json['filename'] = self.filename
    path = self.output_path(self.filename)
    with open(path, 'w') as file:
      json.dump(as_json, file, indent=2, sort_keys=True)

  def invalidate(self):
    self.legend.next_button["state"] = "disabled"

  def output_path(self, filename):
    return OUTPUT_DIR + "/" + filename + ".json"

  def find_existing_json(self):
    path = self.output_path(self.filename)
    if not os.path.isfile(path):
      return None
    with open(path) as json_file:
      data = json.load(json_file)
    return data

  def load(self, path):
    self.path, self.filename = path, Path(path).stem

    with open(path) as file:
      text = file.read()

    self.editor.initialize(text)
    
    existing = self.find_existing_json()
    if existing != None:
      self.editor.load_json(existing)

    self.winfo_toplevel().title("Textinatiratiror: {}".format(self.filename))

  def load_random(self):
    paths = glob.glob(INPUT_DIR + "/*")
    existing_paths = glob.glob(OUTPUT_DIR + "/*")
    for existing in existing_paths:
      existing = INPUT_DIR + "/" + Path(existing).stem + ".txt"
      if existing in paths:
        paths.remove(existing)
    if len(paths) == 0:
      self.editor.initialize("All files are annotated.")
    else:
      self.load(random.choice(paths))

  def load_specific(self):
    path = askopenfilename(initialdir=INPUT_DIR,
        title="Select text file",
        filetypes=(("text files", "*.txt"),))
    if path == tuple():
      return
    self.load(path)

  def delete_current(self):
    path = self.output_path(self.filename)
    if askquestion('Delete annotations',
          'Are you sure you want delete annotations for {}'.format(path),
          icon = 'warning') == 'yes' and \
        os.path.exists(OUTPUT_DIR): 
      os.remove(path)

root = tk.Tk()
root.scale = 1.5
root.tk.call('tk', 'scaling', root.scale)
app = Application(master=root)
app.mainloop()