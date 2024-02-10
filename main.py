import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from interface import HMI
from model import Model


def main():
    model = Model('friday_model')
    hmi = HMI(model)
    try:
        hmi.run()
    except KeyboardInterrupt:
        print("Exiting F.R.I.D.A.Y. process")


if __name__ == "__main__":
    main()
