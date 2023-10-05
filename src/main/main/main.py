from src.main.shell.process import Process
from hmi.interface import HMI
from model.model import Model
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))


def main():
    model = Model('ddd')
    with Process() as msc:
        hmi = HMI(model, msc)
        hmi.run()


if __name__ == "__main__":
    main()
