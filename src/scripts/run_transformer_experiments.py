import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

dataset_dir = os.path.relpath(os.path.join(DATA_FOLDER, 'dataset_conll'), start=PROJECT_FOLDER)

model_name = 'allegro/herbert-base-cased'

max_seq_length = 128
batch_size = 32
num_train_epochs = 1
learning_rate = 5e-5
weight_decay = 0.01


def main():
    os.chdir(PROJECT_FOLDER)

    cmd = 'python -m src.scripts.train_transformer '
    options = [
        '--model_name', model_name,
        '--data_dir', dataset_dir,
        '--batch_size', str(batch_size),
        '--epochs', str(num_train_epochs),
        '--learning_rate', str(learning_rate),
        '--weight_decay', str(weight_decay),
        '--max_seq_length', str(max_seq_length),
        '--do_lower_case',
        '--do_test'
    ]
    cmd += ' '.join(options)

    logger.info(f"Training {model_name}")
    run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
