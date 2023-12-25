import argparse
from threading import BoundedSemaphore
from typing import List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from humanization import config_loader
from humanization.annotations import HeavyChainType
from humanization.humanizer import common_parser_options, Humanizer
from humanization.models import load_model
from humanization.utils import configure_logger

config = config_loader.Config()
logger = configure_logger(config, "Telegram bot")

pool_sema = BoundedSemaphore(value=2)


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    await update.message.reply_text(
        """
        Hi!
        This bot can humanize heavy chains of antibodies. Supported functions:
        - /humanize <v_gene_type> <target_model_metric> <sequence>
        runs humanization tool on given sequence
        <v_gene_type> = 1...7 - target type of v gene
        <target_model_metric> = 0...1 - target model humanization score
        <sequence> - amino acid sequence
        If you have questions write to @Alex2_000
        """
    )


def humanize_gen(humanizers: List[Humanizer]):
    async def humanize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await update.effective_message.reply_text("Your query is in queue")

            with pool_sema:
                v_gene_type = int(context.args[0])
                target_model_metric = float(context.args[1])
                sequence = context.args[2]
                logger.info(f"Execution humanization query from {update.effective_message.from_user.full_name} "
                            f"({update.effective_message.from_user.name}):\n"
                            f"{v_gene_type} {target_model_metric} {sequence}")
                await update.effective_message.reply_text("Humanization process is started")
                humanizer = humanizers[v_gene_type - 1]
                result, iterations = humanizer.query(sequence, target_model_metric)
                iterations_repr = "\n".join(map(str, iterations))
                response = f"Input: {sequence}\n" \
                           f"{iterations_repr}\n" \
                           f"======\n" \
                           f"Result: {result}"
                await update.effective_message.reply_text(response)
        except (IndexError, ValueError):
            await update.effective_message.reply_text("Correct usage specified in /help")

    return humanize


def main(models_dir, dataset_file, modify_cdr,
         skip_positions, deny_use_aa, deny_change_aa, use_aa_similarity) -> None:
    humanizers = []
    for i in range(1, 8):
        model_wrapper = load_model(models_dir, HeavyChainType(str(i)))
        humanizers.append(
            Humanizer(model_wrapper, None, modify_cdr, skip_positions, deny_use_aa, deny_change_aa, use_aa_similarity)
        )

    with open("bot.token", 'r') as file:
        token = file.read()
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler(["start", "help"], start))
    application.add_handler(CommandHandler("humanize", humanize_gen(humanizers)))

    logger.info("Bot polling started")
    application.run_polling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Humanizer telegram bot''')
    common_parser_options(parser)

    args = parser.parse_args()

    main(models_dir=args.models,
         dataset_file=args.dataset,
         modify_cdr=args.modify_cdr,
         skip_positions=args.skip_positions,
         deny_use_aa=args.deny_use_aa,
         deny_change_aa=args.deny_change_aa,
         use_aa_similarity=args.use_aa_similarity)
