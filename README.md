# Antibody humanization

This tool is implementation of humanization based on machine learning methods.
Basic idea has been taken from article (see [References](#references)).

The solution uses CatBoost library.

## Installing

1. Create venv with Python 3.10 using conda.
2. Install anarci.
   ```shell
   cd python/src
   ./install_anarci.sh
   ```
3. Install library and dependencies.
   ```shell
   cd python/src
   pip install --editable .
   ```
4. Download models from GitHub page and extract to the repository root.

If the code cannot find `humanization` library then run the following code in a terminal:
```shell
export PYTHONPATH="$PYTHONPATH:<full-path-to-python/src>"
```

## Scripts:
- [Dataset downloader](#dataset-downloader)
- [Dataset preparer](#dataset-preparer)
- [Classifiers builder](#classifiers-builder)
- [Humanizer](#humanizer)
- [Configuration](#configuration)
- [Testing](#testing)
 
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
usage: dataset_reader.py [-h] [--schema SCHEMA] [--skip-existing]
                         [--process-existing]
                         input {H,L} output

Dataset preparer

positional arguments:
  input               Path to input folder with .csv files
  {H,L}               Chain kind
  output              Path to output folder

options:
  -h, --help          show this help message and exit
  --schema SCHEMA     Annotation schema
  --skip-existing     Skip existing processed files
  --process-existing
```

Example:
```shell
python3 dataset_reader.py bash/raw_dataset H bash/annotated_dataset
```

### Classifiers builder

Training models of all 7 V gene types.

```
usage: heavy_random_forest.py [-h] [--iterative-learning]
                              [--single-batch-learning] [--schema {chothia}]
                              [--metric {youdens,matthews}]
                              [--tree-lib {catboost,sklearn}]
                              [--print-metrics] [--types TYPES]
                              input output

Heavy chain RF generator

positional arguments:
  input                 Path to directory where all .csv (or .csv.gz) are
                        listed
  output                Output models location

options:
  -h, --help            show this help message and exit
  --iterative-learning  Iterative learning using data batches
  --single-batch-learning
  --schema {chothia}    Annotation schema
  --metric {youdens,matthews}
                        Threshold optimized metric
  --tree-lib {catboost,sklearn}
                        Decision tree library
  --print-metrics       Print learning metrics
  --types TYPES         Build only specified types
```

Example:
```shell
python3 heavy_random_forest.py bash/annotated_dataset models --schema chothia
```

Animal models could be retrieved in the same way using `animal_random_forest.py`.

### Humanizer

Greedy humanizer.
Takes the most effective aminoacids from a reference and inserts to the given sequence.
It continues while V-gene score >= 0.85 is not achieved.
Change choice relies on complex penalty containing humanness score, embedding distance and distance to animal reference.

```
usage: greedy_humanizer.py [-h] [--input INPUT] [--output OUTPUT]
                           [--models MODELS] [--human-sample HUMAN_SAMPLE]
                           [--human-chain-type HUMAN_CHAIN_TYPE]
                           [--dataset DATASET] [--wild-dataset WILD_DATASET]
                           [--deny-use-aa DENY_USE_AA]
                           [--deny-change-aa DENY_CHANGE_AA]
                           [--deny-change-pos DENY_CHANGE_POS]
                           [--change-batch-size CHANGE_BATCH_SIZE]
                           [--limit-changes LIMIT_CHANGES]
                           [--candidates-count CANDIDATES_COUNT]
                           [--report REPORT]

Greedy antiberta humanizer

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to input fasta file
  --output OUTPUT       Path to output fasta file
  --models MODELS       Path to directory with random forest models
  --human-sample HUMAN_SAMPLE
                        Human sample used for creation chimeric sequence
  --human-chain-type HUMAN_CHAIN_TYPE
                        Type of provided human sample
  --dataset DATASET     Path to dataset for humanness calculation
  --wild-dataset WILD_DATASET
                        Path to dataset for wildness calculation
  --deny-use-aa DENY_USE_AA
                        Amino acids that could not be used
  --deny-change-aa DENY_CHANGE_AA
                        Amino acids that could not be changed
  --deny-change-pos DENY_CHANGE_POS
                        Positions that could not be changed (fwr1_12, fwr2_2,
                        etc.)
  --change-batch-size CHANGE_BATCH_SIZE
                        Count of changes that will be applied in one iteration
  --limit-changes LIMIT_CHANGES
                        Limit count of changes
  --candidates-count CANDIDATES_COUNT
                        Count of used references
  --report REPORT       Path to report file
```

Example:
```shell
python3 greedy_humanizer.py models --input in.fasta --output out.fasta --models ../../../../sklearn_models2
```

### Configuration

There is a bunch of configurable settings in `config.yaml` file.

### Testing

In folder `testing` there are few scripts for drawing plots:

```
ada_analyzer.py
embedding_analyzer.py
feature_importance.py
humanness_calculator_2.py
```

They need to be run from the directory without any arguments:

```shell
cd testing
python3 ada_analyzer.py
```

## References

1. https://academic.oup.com/bioinformatics/article/37/22/4041/6295884
