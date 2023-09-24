import tkinter
from ..shell import MCS

class HMI:
    '''
        This class is the GUI of the chatbot.
    '''
    def __init__(self, model) -> None:
        self.__model = model
        pass

    @property
    def model(self):
        return self.__model
    
    def run(self) -> None:
        pass

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