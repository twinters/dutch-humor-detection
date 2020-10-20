import argparse

import pandas as pd
from pycm import ConfusionMatrix


def construct_arg_parser():
    parser = argparse.ArgumentParser(description='Calculate cm from labels and predictions')
    parser.add_argument('labels', metavar='labels', type=argparse.FileType('r'), )
    parser.add_argument('predictions', metavar='predictions', type=argparse.FileType('r'), )

    parser.add_argument('-o', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = construct_arg_parser()

    df = pd.DataFrame(args.labels, columns=["labels"])
    df['labels'] = df['labels'].astype(bool)
    df['predictions'] = pd.DataFrame(args.predictions, columns=['predictions'])['predictions'].str.startswith("1")

    print(ConfusionMatrix(actual_vector=df['labels'].values, predict_vector=df['predictions'].values))


