import argparse

from humanization.common import config_loader
from humanization.common.utils import configure_logger
from humanization.external_models.ablang_utils import get_attentions

config = config_loader.Config()
logger = configure_logger(config, "Group analyzer")


def main(group_size, only_changes):
    while True:
        original_seq = input("Wild > ")
        humanized_seq = input("Hum >  ")
        attentions = get_attentions(original_seq)
        length = len(original_seq)
        print("       " + original_seq)
        humanized_changes = [False] * length
        if humanized_seq != "":
            humanized_changes = [humanized_seq[i] != original_seq[i] for i in range(length)]
            print("       " + ''.join('X' if ch else '.' for ch in humanized_changes))
        for i in range(length):
            if only_changes and not humanized_changes[i]:
                continue
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
    parser.add_argument('--only-changes', type=bool, required=False, default=True, help='Show only changes groups')
    args = parser.parse_args()
    main(args.size, args.only_changes)
