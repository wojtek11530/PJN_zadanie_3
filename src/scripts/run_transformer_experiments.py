import logging
import os
import sys

from src.settings import DATA_FOLDER
from src.transformer_models.transformer_model_utils import train_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

dataset_dir = os.path.join(DATA_FOLDER, 'dataset_conll')

model_names = ['allegro/herbert-base-cased', 'clarin-pl/roberta-polish-kgr10']
output_size = 4
max_seq_length = 128
batch_size = 32
num_train_epochs = 3
learning_rate = 5e-5
weight_decay = 0.01


def main():
    for model_name in model_names:
        args = {
            'model_name': model_name,
            'data_dir': dataset_dir,
            'output_size': output_size,
            'batch_size': batch_size,
            'epochs': num_train_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'max_seq_length': max_seq_length,
            'do_lower_case': True,
            'do_test': True
        }
        logger.info(f"Training {model_name}")
        train_model(**args)


if __name__ == '__main__':
    main()
