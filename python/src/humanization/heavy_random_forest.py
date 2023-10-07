import argparse

import config_loader
from humanization import abstract_random_forest
from humanization.abstract_random_forest import configure_abstract_parser
from humanization.annotations import GeneralChainType
from humanization.models import save_model
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Heavy chain RF")


def main(input_dir, schema, metric, output_dir, annotated_data, iterative_learning, print_metrics, tree_types):
    def make_models(chain_type: GeneralChainType):
        return abstract_random_forest.make_models(
            input_dir, annotated_data, schema, chain_type, metric, iterative_learning, print_metrics, tree_types
        )
    for wrapped_model in make_models(GeneralChainType.HEAVY):
        save_model(output_dir, wrapped_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Heavy chain RF generator''')
    configure_abstract_parser(parser)
    args = parser.parse_args()
    main(input_dir=args.input, schema=args.schema, metric=args.metric, output_dir=args.output,
         annotated_data=args.annotated_data, iterative_learning=args.iterative_learning,
         print_metrics=args.print_metrics, tree_types=args.types)
