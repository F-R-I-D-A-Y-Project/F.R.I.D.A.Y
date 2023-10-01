import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import tkinter as tk
from shell.msc import MSC
from model.model import Model

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class HMI:
    '''
        This class is the GUI of the chatbot.
    '''
    def __init__(self, model) -> None:
        self.__model = model
        self.__gui = tk.Tk()
        self.initialize()

    @property
    def model(self):
        return self.__model    

    @property
    def gui(self):
        '''
        
        '''
        return self.__gui
    
    def initialize(self):
        '''
            This method initializes the GUI.
        '''
        self.gui.title("Chat")
        self.gui.resizable(width=False, height=False)
        self.gui.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = tk.Label(self.gui, bg=BG_COLOR, fg=TEXT_COLOR,
                              text="F.R.I.D.A.Y",
                              font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = tk.Label(self.gui, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_area = tk.Text(self.gui, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_area.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_area.configure(cursor="arrow", state=tk.DISABLED)

        # scroll bar
        scrollbar = tk.Scrollbar(self.text_area)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_area.yview)

        # bottom label
        bottom_label = tk.Label(self.gui, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.text_box = tk.Text(bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.text_box.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.text_box.focus()
        self.text_box.bind("<Return>", self.send)

        # send button
        self.__button = tk.Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                                  command=self.send)
        self.__button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        
        # message entry box
        self.entry = tk.Entry(bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.entry.focus()
        self.entry.bind("<Return>", self.send)

    def run(self) -> None:
        '''
            This method runs the GUI.
        '''
        self.gui.mainloop()

    def send(self, event=None) -> None:
        '''
            This method sends a message to the chatbot.
        '''
        message = self.entry.get()
        if not message:
            return
        self.text_box.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, "You: " + message + '\n\n')
        self.text_area.insert(tk.END, "F.R.I.D.A.Y: " + self.answer_to(message) + '\n\n')
        pass

    def answer_to(self, message: str) -> str:
        '''
            This method returns the answer of the chatbot to a message.
        '''
        
        # answer: str = self.model.predict(message)
        # if answer.startswith("@command: "):
        #     with MCS() as mcs:
        #         answer = mcs.send(answer[10:])
        
        return '?'