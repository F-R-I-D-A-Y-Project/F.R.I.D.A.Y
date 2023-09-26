from ..model.model import Model
from ..hmi.interface import HMI
from ..shell.msc import MCS

def main():
    model = Model()
    hmi = HMI(model)
    hmi.run()