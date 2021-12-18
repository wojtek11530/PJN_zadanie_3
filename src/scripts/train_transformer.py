import argparse
import logging
import sys

from src.transformer_models.transformer_model_utils import train_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The data dir with files hotels.sentence.train.txt, "
                             "hotels.sentence.dev.txt, hotels.sentence.test.txt")
    parser.add_argument('--model_name',
                        default=None,
                        type=str,
                        required=True,
                        help='Name of Hugging face model')
    parser.add_argument("--output_size",
                        default=4,
                        type=int,
                        help="The output size of model, i.e. classes number")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximal sequence length")
    parser.add_argument('--do_lower_case',
                        action='store_true')

    parser.add_argument("--epochs",
                        default=3,
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=0.01,
                        type=float,
                        metavar='W',
                        help='weight decay')

    parser.add_argument('--do_test',
                        action='store_true',
                        help='Performing evaluation on test set after training')

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    train_model(**vars(args))


if __name__ == '__main__':
    main()
