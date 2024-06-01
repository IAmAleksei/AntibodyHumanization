import argparse
import pandas as pd


def main(filename):
    df = pd.read_csv(filename)
    types = ["Wild.", "Therap.", "i", "i3", "HuMab", "Sapiens1", "Sapiens3"]
    for t in types:
        print(t)
        state0 = df["Type"] == t
        state1 = df["HuVGS"] >= 0.845
        state2 = df["HuVGS"] > df["WVGS"]
        print("Total", df[state0].shape[0])
        print("HuVGS >= 0.85", df[state0 & state1].shape[0])
        print("HuVGS > WVGS", df[state0 & state2].shape[0])
        print("Satisfied all criteria", df[state0 & state1 & state2].shape[0])
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Comparator of sequences''')
    parser.add_argument('filename', type=str, help='Input .csv')
    args = parser.parse_args()
    main(args.filename)
