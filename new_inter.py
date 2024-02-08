import customtkinter as ctk
from model import Model


BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

GPT_LEFT_TAB = "#202123"
GPT_BOT_ANSWER = "#343541"
GPT_USER_INPUT = "#444654"
GPT_TEXT_BOX = "#40414F"


FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class HMI:

    def __init__(self, model: Model) -> None:
        self.__model = model
        self.__answer = ''
        self.__gui = ctk.CTk()
        self.initialize()

    model, gui = property(lambda self: self.__model), property(
        lambda self: self.__gui)

    def initialize(self):
        self.gui.geometry("750x550")
        self.gui.title("Chat")
        self.gui.resizable(width=False, height=False)
        ctk.set_default_color_theme("dark-blue")
        title_label = ctk.CTkLabel(
            self.gui, text="F.R.I.D.A.Y.", font=ctk.CTkFont(FONT_BOLD, size=30, weight='bold'))
        title_label.pack(pady=10, padx=10)

        self.frame = ctk.CTkFrame(self.gui, bg_color=BG_COLOR,
                             width=750, height=500, fg_color=GPT_TEXT_BOX)
        self.text_widget = ctk.CTkTextbox(self.frame, width=750, height=420, bg_color=GPT_TEXT_BOX,
                                  fg_color=GPT_TEXT_BOX, font=ctk.CTkFont(FONT, 14), state='disabled')
        self.text_widget.pack(padx=30)
        self.text_entry = ctk.CTkEntry(self.frame, width=700, bg_color=GPT_USER_INPUT,height=50, fg_color=GPT_LEFT_TAB, font=ctk.CTkFont(FONT, 14), placeholder_text="Ask anything to F.R.I.D.A.Y.")
        self.text_entry.bind("<Return>", lambda e: self.send())
        self.text_entry.pack(padx=30, pady=30)
        self.frame.pack()

    def send(self):
        msg = self.text_entry.get()
        self.text_entry.delete(0, ctk.END)

        if msg != '':
            self.text_widget.configure(state="normal")
            self.text_widget.insert(ctk.END, "You: " + msg + '\n')
            self.text_widget.configure(state="disabled")

            self.answer = self.model(msg)
            self.text_widget.configure(state="normal")
            self.text_widget.insert(
                ctk.END, "F.R.I.D.A.Y.: " + self.answer + '\n')
            self.text_widget.configure(state="disabled")

            self.text_widget.see(ctk.END)

    def run(self):
        self.gui.mainloop()


