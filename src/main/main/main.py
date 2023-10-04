import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from model.model import Model
from hmi.interface import HMI
from shell.msc import MSC

def main():
    model = Model('ddd')
    hmi = HMI(model)
    hmi.run()

if __name__ == "__main__":
    main()