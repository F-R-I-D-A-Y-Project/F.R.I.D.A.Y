import torch.nn as nn
import torch
import torch.nn.functional as F
import torchtext
import math
import warnings
import csv, pickle
import pathlib
import subprocess
from typing import Self
warnings.simplefilter("ignore")



class Model:
    '''
        This class is responsible for the NLP model of the chatbot.
    '''
    def __init__(self: Self, path_to_dataset: str) -> None:
        '''
            Constructor of the class. It receives the path to the dataset, but does not train the model.
            
            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        self.__dataset = path_to_dataset
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self: Self, path_to_dataset: str|None=None, *, epochs: int=100, verbose: bool=True) -> None:
        '''
            This method is responsible for training the model.
            It reads the dataset, captures the amount of unique words existent in the dataset, 
            creates the Transformers Model and trains it

            If the dataset was unchanged, it loads the model from the pickle file. If not, the training algorithm will be executed.

            Args:
                path_to_dataset (str): path to dataset used for training
        '''
        if path_to_dataset:
            self.__dataset = path_to_dataset

        if (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').exists():
            with open('model.pkl', 'rb') as f:
                self.__model = pickle.load(f)
        else:
            self.__train(epochs, verbose)

    def __serialize_model(self: Self) -> None:
        '''
            Serializes the model into a pickle file to avoid retraining it every time the chatbot is executed.
        '''

        with (pathlib.Path(__file__).parent.parent.parent.parent / 'model.pkl').open('wb') as f:
            pickle.dump(self.__model, f)

    def __train(self: Self, epochs: int, verbose: bool) -> None:
        '''
            Training algorithm of the Tranformers model
        '''

        with open(self.__dataset, 'r') as f:
            data = csv.reader(f)

        self.__model = None # Transformer(512, 10000, 10000, 100).to(self.__device)

        for epoch in range(epochs):
            self.__model.train()
            ...
        if verbose:
            print(self.__model.parameters())
        
        self.__serialize_model()

    def predict(self: Self, message: str) -> str:
        '''
            This method is responsible for returning the answer of the model for the chatbot.
            It receives a message, tokenizes it, and passes it to the Transformers Model

            Args:
                message (str): message to be answered by the chatbot

        '''
        trg = None #?
        return self.__model.decode(message, trg)

    __call__ = predict
    
    def check_db_change(self: Self, commit_on_change: bool=True) -> None:
        '''
            Verifies changes in the dataset. If there are changes, it will delete the serialized model.

            OBS: The changes are verified via git, so in order to properly verify the difference, commits will be made
            every time 
        '''
        out = subprocess.run(f'git diff {self.__dataset}/../approved.csv', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                       capture_output=True).stdout.strip()
        if out:
            subprocess.run(f'rm -f model.pkl', shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())
            self.__update_dataset()
            if commit_on_change:
                subprocess.run([f'git add .', 'git commit -am "Update dataset" --amend'], shell=True, cwd=pathlib.Path(__file__).parent.parent.parent.parent.absolute())

    def __update_dataset(self: Self) -> None: pass