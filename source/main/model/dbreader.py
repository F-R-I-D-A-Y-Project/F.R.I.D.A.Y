import pathlib
from typing import Self, Callable
from torch.utils.data import Dataset, DataLoader


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