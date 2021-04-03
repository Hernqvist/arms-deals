import tkinter as tk

class Label:
  def __init__(self, start, end, tag):
    self.start = start
    self.end = end
    self.tag = tag

def n_chars(n):
  return '1.0 + {} chars'.format(n)

class Editor(tk.Text):
  def __init__(self, master=None):
    super().__init__(master=master)
    self.master = master

    text = ""
    for i in range(50):
      text = text + "hello this is text {}\n".format(i)
    self.initialize(text)

  def refresh(self):
    self.windows = []
    self.config(state='normal')
    for tag in self.tag_names():
      self.tag_delete(tag)

    for window in self.windows:
      window.destroy()  

    if self.text != self.old_text:
      self.delete('1.0', 'end')
      self.insert('1.0', self.text)
      self.old_text = self.text

    for i, label in enumerate(self.labels):
      tag_name = 'label_{}'.format(i)
      self.mark_set(tag_name + '_start', n_chars(label.start))
      self.mark_set(tag_name + '_end', n_chars(label.end))

      self.tag_add(tag_name, tag_name + '_start', tag_name + '_end')
      self.tag_configure(tag_name, background='yellow')

    for i, label in enumerate(self.labels):
      tag_name = 'label_{}'.format(i)
      b = tk.Button(self, text="Mark!")
      self.window_create(tag_name + '_start', window=b)
      self.windows.append(b)

    self.config(state='disabled')

  def add_label(self, label):
    self.labels.append(label)
    self.refresh()

  def initialize(self, text):
    self.text = text
    self.old_text = ""
    self.labels = []
    self.refresh()

  def key_pressed(self, key):
    start, end = map(self.to_integer_index, self.tag_ranges('sel'))
    self.add_label(Label(start, end, "Answer"))

  def to_integer_index(self, text_index):
    index = 0
    row, col = map(int, str(text_index).split('.'))
    print(row, col)
    for i, line in enumerate(self.text.split('\n')):
      i += 1
      if i < row:
        index += len(line) + 1
      else:
        index += col
        break
    print(index)
    return index

class Application(tk.Frame):

  def key_pressed(self, event):
    self.editor.key_pressed(event.char)

  def __init__(self, master=None):
    super().__init__(master)
    self.master = master
    self.pack(expand=True, fill='both')
    self.editor = Editor(self)
    self.editor.pack(expand=True, fill='both')
    self.master.bind("<Key>", self.key_pressed)

root = tk.Tk()
#root.tk.call('tk', 'scaling', 1.0)
app = Application(master=root)
app.mainloop()