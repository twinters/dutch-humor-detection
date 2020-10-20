import argparse
import json
import os
import random
import sys

import pandas as pd


def classification(args):
    df = pd.DataFrame(json.load(args.fileA), columns=["text"])
    df['label'] = 0

    df2 = pd.DataFrame(json.load(args.fileB), columns=['text'])
    df2['label'] = 1

    combined = df.append(df2, ignore_index=True)
    combined = combined.sample(frac=1.0)  # Shuffle

    write(args, combined)


def ranked(args):
    df = pd.DataFrame(json.load(args.fileA), columns=["original"])
    df['scrambled'] = pd.DataFrame(json.load(args.fileB), columns=['scrambled'])['scrambled']

    df = df.sample(frac=1.0)  # Shuffle

    df['text'] = pd.Series()
    df['label'] = pd.Series()
    for row in df.iterrows():
        random_position = random.random() > 0.5

        df['text'][row[0]] = row[1].original + " <sep> " + row[1].scrambled if random_position else row[1].scrambled + " <sep> " + \
                                                                                      row[1].original

        df['label'][row[0]] = "1" if random_position else "0"
    write(args, df)

def write(args, df):
    if not args.o:
        for row in df.iterrows():
            print(row[1].text.replace("\n", "\\n"))
            print(row[1].label, file=sys.stderr)
    else:
        train_idx = int(len(df) * 0.75)
        dev_idx = int(len(df) * 0.85)

        train = df[:train_idx]
        dev = df[train_idx:dev_idx]
        test = df[dev_idx:]

        for name, data in {'train': train, 'dev': dev, 'test': test}.items():
            with open(os.path.join(args.o, name + ".sentences"), mode="w") as sfp, \
                    open(os.path.join(args.o, name + ".labels"), mode="w") as lfp:
                for row in data.iterrows():
                    print(row[1].text.replace("\r\n", "\\n").replace("\n", "\\n"), file=sfp)
                    print(row[1].label, file=lfp)

def construct_arg_parser():
    parser = argparse.ArgumentParser(description='Process 2 json files')
    parser.add_argument('fileA', metavar='A', type=argparse.FileType('r'), )
    parser.add_argument('fileB', metavar='B', type=argparse.FileType('r'), )
    parser.add_argument('--mode', dest='mode', action='store',
                        default="classification",
                        help='"classification" or "ranked"')

    parser.add_argument('-o', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = construct_arg_parser()

    print(args)
    if args.mode == "classification":
        classification(args)
    elif args.mode == "ranked":
        ranked(args)
