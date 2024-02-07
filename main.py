import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from interface import HMI
from model import Model


def main():
    model = Model('friday_model', 'friday_model_tokenizer')
    hmi = HMI(model)
    try:
        hmi.run()
    except KeyboardInterrupt:
        print("Exiting F.R.I.D.A.Y. process")


if __name__ == "__main__":
    main()
