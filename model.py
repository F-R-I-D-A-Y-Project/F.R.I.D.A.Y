from typing import Self
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pathlib
import subprocess
import logging
import time

logging.basicConfig(level=logging.WARNING,
                    format='\033[1;33m[%(levelname)s] %(message)s\033[0m')


class Model:
    '''
        This class is responsible for the NLP model of the chatbot.
    '''

    def __init__(self: Self, path_to_model: str, path_to_tokenizer: str | None = None, maxlen: int = 100) -> None:
        if not pathlib.Path(path_to_model).exists():
            logging.warning('Model not found. Generating model...')
            time.sleep(1)
            subprocess.run('python model_gen.py', shell=True)
        self.__model = AutoModelForCausalLM.from_pretrained(path_to_model)
        self.__tokenizer = AutoTokenizer.from_pretrained(
            path_to_tokenizer or path_to_model)
        self.__pipe = pipeline('text-generation', model=self.__model,
                               tokenizer=self.__tokenizer, max_length=maxlen)

    @property
    def model(self: Self):
        return self.__model

    @property
    def tokenizer(self: Self):
        return self.__tokenizer

    @property
    def pipe(self: Self):
        return self.__pipe

    def predict(self: Self, message: str) -> str:
        '''
            This method is responsible for returning the answer of the model for the chatbot.
            It receives a message, tokenizes it, and passes it to the Transformers Model

            Args:
                message (str): message to be answered by the chatbot

        '''
        prompt = "<startofstring> " + message + " <bot>: "
        ret = self.pipe(prompt)[0]['generated_text']
        return ret.split("<bot>: ")[1].strip()

    __call__ = predict
