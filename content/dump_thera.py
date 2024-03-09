import argparse
import json


def dump(fasta_output):
    mapper = {
        "sequ": "Wild",
        "ther": "Therap",
        "hu_m": "HuMab",
        "sap1": "Sapiens1",
        "sap3": "Sapiens3"
    }
    with open('thera_antibodies.json', 'r') as fp:
        samples = json.load(fp)
    with open('extra_thera_antibodies.json', 'r') as fp:
        samples += json.load(fp)
    with open(fasta_output, 'w') as file:
        for antibody in samples:
            for tp in mapper.keys():
                file.write(f"> {antibody['name']}__{mapper[tp]}\n")
                file.write(antibody['heavy'][tp].replace('-', '') + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Dumper of antibody data''')
    parser.add_argument('--fasta-output', type=str, default="db.fasta", required=False, help='Path to output file')
    args = parser.parse_args()
    dump(fasta_output=args.fasta_output)
