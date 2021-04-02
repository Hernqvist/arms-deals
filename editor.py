import tkinter as tk

class Editor(tk.Text):
  def __init__(self, master=None):
    super().__init__(master=master)
    self.master = master
    for i in range(50):
      self.insert('end', "{}hello this is text\n".format(i))
    self.tag_add('highlightline', '5.0', '6.0')
    self.tag_configure('highlightline', background='yellow')

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(expand=True, fill='both')
        self.editor = Editor(self)
        self.editor.pack(expand=True, fill='both')

root = tk.Tk()
#root.tk.call('tk', 'scaling', 1.0)
app = Application(master=root)
app.mainloop()