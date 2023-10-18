# from torchdata.datapipes import DataChunk 
import pathlib
class DBManager:
    def __init__(self, path: pathlib.Path) -> None:
        self.__dataset_path = path

    @property
    def dataset_path(self) -> pathlib.Path:
        '''
            This property returns the path to the dataset.
        '''
        return self.__dataset_path
    
    