import argparse

from humanization.common import config_loader
from humanization.common.utils import configure_logger
from humanization.external_models.ablang_utils import get_attentions

config = config_loader.Config()
logger = configure_logger(config, "Group analyzer")


def main(group_size):
    while True:
        seq = input(">     ")
        attentions = get_attentions(seq)
        length = len(seq)
        for i in range(length):
            res_attn = sorted(enumerate(attentions[i]), key=lambda x: (x[1], x[0]), reverse=True)
            ans = ["."] * length
            for j in range(group_size):
                pos = res_attn[j][0]
                ans[pos] = str(j)
            ans[i] = "*"
            print(f"{i:03d}#. {''.join(ans)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Group analyzer of sequences''')
    parser.add_argument('--size', type=int, required=False, default=5, help='Group size')
    args = parser.parse_args()
    main(args.size)
