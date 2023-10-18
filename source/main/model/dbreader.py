import pathlib
from typing import Self, Callable


class DBIterator:
    def __init__(self: Self, operation: Callable, *args, **kwargs) -> None:
        self.__operation = operation
        self.__args = args
        self.__kwargs = kwargs    

    def collect(self): 
        return self.__operation(*self.__args, **self.__kwargs)


class DBManager:
    def __init__(self, path: pathlib.Path,
                 dataset: str='dataset.csv', 
                 command_dataset: str='commands.csv') -> None:
        self.__dataset_path = path

    @property
    def dataset_path(self) -> pathlib.Path:
        '''
            This property returns the path to the dataset.
        '''
        return self.__dataset_path
    
    def split(self: Self, train_size: float=0.8) -> None:
        '''
            This method splits the dataset into train, test and validation sets.
        '''
        return DBIterator(self.__split, train_size)
    
    def __split(self, train_size):
        '''
            This method splits the dataset into train, test and validation sets.
        '''