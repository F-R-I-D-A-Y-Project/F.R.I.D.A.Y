import tkinter as tk
from ..shell.msc import MSC
from ..model.model import Model

class HMI:
    '''
        This class is the GUI of the chatbot.
    '''
    def __init__(self, model) -> None:
        self.__model = model
        self.__gui = tk.Frame()
        self.__button = tk.Button()
        self.__text_area = ...
        self.__text_box = ...

    @property
    def model(self):
        return self.__model    

    @property
    def gui(self):
        '''
        
        '''
        return self.__gui
    
    @property
    def send_button(self) -> None:
        '''

        '''
        pass

    @property
    def text_area(self):
        '''

        '''
        pass

    @property
    def text_box(self):
        '''
        
        '''
        pass

    def run(self) -> None:
        '''
        
        '''
        self.gui.mainloop()

    def answer_to(self, message: str) -> str:
        '''
            This method returns the answer of the chatbot to a message.
        '''
        return '?'
        # answer: str = self.model.predict(message)
        # if answer.startswith("@command: "):
        #     with MCS() as mcs:
        #         answer = mcs.send(answer[10:])
    pass