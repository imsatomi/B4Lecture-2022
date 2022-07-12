import argparse

from model_maker import ModelMaker

SAMPLERATE = 8000

INFO_FILE = "keras_model/my_model_info.txt"
EST_FILE = "keras_model/my_model.h5"
LR = 1e-4
DATA_SIZE = 4096
BATCH_SIZE = 16
EPOCH = 30
VALID_RATE = 0.2
N_MFCC = 13


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mfcc", type=int, default=13, help="dimension of MFCC")
    parser.add_argument("--epoch", type=int, default=30, help="epoch size")
    args = parser.parse_args()
    N_MFCC = args.n_mfcc
    EPOCH = args.epoch

    maker = ModelMaker(
        samplerate=SAMPLERATE,
        info_file=INFO_FILE,
        est_file=EST_FILE,
        lr=LR,
        data_size=DATA_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        valid_rate=VALID_RATE,
        n_mfcc=N_MFCC,
    )

    maker.execute()


if __name__ == "__main__":
    main()
