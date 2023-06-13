# Antibody humanization

This tool is implementation of humanization based on machine learning methods.
Basic idea has been taken from article (see [References](#references)).

The solution uses CatBoost library.

## Modules:
- [Dataset downloader](#dataset-downloader)
- [Dataset preparer](#dataset-preparer)
- [Classifiers builder](#classifiers-builder)
- [Humanizer](#humanizer)
- [Reverse humanizer](#reverse-humanizer)
- [Telegram bot](#telegram-bot)
 
### Dataset downloader

Scripts for downloading all IGHG chains from OAS.

```shell
./heavy_download.sh destination-path
```

Example:
```shell
./heavy_download.sh raw_dataset
```

### Dataset preparer

Annotating sequences using specified annotation.

```
dataset_preparer.py [-h] [--skip-existing] [--process-existing]
                           input schema output

positional arguments:
  input               Path to input folder with .csv files
  schema              Annotation schema
  output              Path to output folder

optional arguments:
  -h, --help          show this help message and exit
  --skip-existing     Skip existing processed files
  --process-existing
```

Example:
```shell
python3 dataset_preparer.py bash/raw_dataset chothia bash/annotated_dataset
```

### Classifiers builder

Training models of all 7 V gene types.

```
heavy_random_forest.py [-h] [--annotated-data] [--raw-data]
                              [--schema SCHEMA] [--metric METRIC]
                              input output

positional arguments:
  input             Path to directory where all .csv (or .csv.gz) are listed
  output            Output models location

optional arguments:
  -h, --help        show this help message and exit
  --annotated-data  Data is annotated
  --raw-data
  --schema SCHEMA   Annotation schema
  --metric METRIC   Threshold optimized metric
```

Example:
```shell
python3 heavy_random_forest.py bash/annotated_dataset models --schema chothia
```

### Humanizer

Direct transformation an animal chain sequence.
Grafting method takes every single amino acid in sequence and tries to change it with best humanizing effect.

```
humanizer.py [-h] [--input INPUT] [--output OUTPUT]
                    [--skip-positions SKIP_POSITIONS] [--dataset DATASET]
                    [--annotated-data] [--raw-data] [--use-aa-similarity]
                    [--ignore-aa-similarity] [--modify-cdr] [--skip-cdr]
                    [--deny-use-aa DENY_USE_AA]
                    [--deny-change-aa DENY_CHANGE_AA]
                    models

positional arguments:
  models                Path to directory with models

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to input fasta file
  --output OUTPUT       Path to output fasta file
  --skip-positions SKIP_POSITIONS
                        Positions that could not be changed
  --dataset DATASET     Path to dataset for humanness calculation
  --annotated-data      Data is annotated
  --raw-data
  --use-aa-similarity   Use blosum table while search best change
  --ignore-aa-similarity
  --modify-cdr          Allow CDR modifications
  --skip-cdr            Deny CDR modifications
  --deny-use-aa DENY_USE_AA
                        Amino acids that could not be used
  --deny-change-aa DENY_CHANGE_AA
                        Amino acids that could not be changed
```

Example:
```shell
python3 humanizer.py models --input input.fasta --dataset bash/annotated_dataset --skip-cdr
```

### Reverse humanizer

Reversed version of humanizer.
Tool combines animal CDR segments and suitable human FR segments, creating chimeric antibody.
Then grafting method takes every single amino acid in sequence and tries to change to animal AA, keeping in the mind humanization score.

```
reverse_humanizer.py [-h] [--skip-positions SKIP_POSITIONS]
                            [--dataset DATASET] [--annotated-data]
                            [--raw-data] [--use-aa-similarity]
                            [--ignore-aa-similarity] [--input INPUT]
                            [--output OUTPUT] [--human-sample HUMAN_SAMPLE]
                            models

positional arguments:
  models                Path to directory with models

optional arguments:
  -h, --help            show this help message and exit
  --skip-positions SKIP_POSITIONS
                        Positions that could not be changed
  --dataset DATASET     Path to dataset for humanness calculation
  --annotated-data      Data is annotated
  --raw-data
  --use-aa-similarity   Use blosum table while search best change
  --ignore-aa-similarity
  --input INPUT         Path to input fasta file
  --output OUTPUT       Path to output fasta file
  --human-sample HUMAN_SAMPLE
                        Human sample used for creation chimeric sequence
```

Example:
```shell
python3 reverse_humanizer.py models --input input.fasta --dataset bash/annotated_dataset
```

### Telegram bot

Humanizer bot.

```
bot.py [-h] [--skip-positions SKIP_POSITIONS] [--dataset DATASET]
              [--annotated-data] [--raw-data] [--use-aa-similarity]
              [--ignore-aa-similarity] [--modify-cdr] [--skip-cdr]
              [--deny-use-aa DENY_USE_AA] [--deny-change-aa DENY_CHANGE_AA]
              models

positional arguments:
  models                Path to directory with models

optional arguments:
  -h, --help            show this help message and exit
  --skip-positions SKIP_POSITIONS
                        Positions that could not be changed
  --dataset DATASET     Path to dataset for humanness calculation
  --annotated-data      Data is annotated
  --raw-data
  --use-aa-similarity   Use blosum table while search best change
  --ignore-aa-similarity
  --modify-cdr          Allow CDR modifications
  --skip-cdr            Deny CDR modifications
  --deny-use-aa DENY_USE_AA
                        Amino acids that could not be used
  --deny-change-aa DENY_CHANGE_AA
                        Amino acids that could not be changed
```

Example:
```shell
python3 bot.py models --dataset bash/annotated_dataset --skip-cdr
```

## References

1. https://academic.oup.com/bioinformatics/article/37/22/4041/6295884
