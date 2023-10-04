import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import tkinter as tk
from shell.msc import MSC
from model.model import Model

BG_GRAY = "#ABB2B9"
BG_GRAY_2 = "#444654" # chatGPT
BG_COLOR = "#17202A"
BG_COLOR_2 =  "#343541" # chatGPT
BG_COLOR_3 = "#202123" # chatGPT
TEXT_COLOR = "#EAECEE"
TEXT_COLOR_2 = "#40414F" # chatGPT


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
        self.gui.resizable(width=True, height=True)
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

        # message box
        bottom_label2 = tk.Label(bottom_label, bg=BG_COLOR, height=80)
        bottom_label2.place(relwidth=0.74 + 0.24, relheight=0.06, rely=0.008, relx=0.011)


        # message entry box
        self.text_box = tk.Entry(bottom_label2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, relief=tk.FLAT, highlightthickness=0, borderwidth=0)
        self.text_box.place(relwidth=0.9, relheight=1, rely=0, relx=0)
        self.text_box.focus()
        self.text_box.bind("<Return>", self.send)

        # send button
        self.__button = tk.Button(bottom_label2, text="Send", font=FONT_BOLD, width=16, bg=BG_GRAY,
                                  command=self.send, relief=tk.FLAT)
        self.__button.place(relx=0.9, rely=0.2, relheight=0.6, relwidth=0.1)
        
    def run(self) -> None:
        '''
            This method runs the GUI.
        '''
        self.gui.mainloop()

    def send(self, event=None) -> None:
        '''
            This method sends a message to the chatbot.
        '''
        message = self.text_box.get()
        if not message:
            return
        self.text_box.delete(0, tk.END)
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, "You: " + message + '\n\n')
        self.text_area.insert(tk.END, "F.R.I.D.A.Y: " + self.answer_to(message) + '\n\n')
        self.text_area.configure(state=tk.DISABLED)

    def answer_to(self, message: str) -> str:
        '''
            This method returns the answer of the chatbot to a message.
        '''
        
        # answer: str = self.model.predict(message)
        # if answer.startswith("@command: "):
        #     with MCS() as mcs:
        #         answer = mcs.send(answer[10:])
        
        return '?'