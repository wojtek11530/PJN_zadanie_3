import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data.dataset_transformer import TextDatasetForTransformers
from src.settings import MODELS_FOLDER
from src.utils import dictionary_to_json, is_folder_empty, get_confusion_matrix_plot

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(model_name: str, data_dir: str, output_size: int,
                epochs: int, batch_size: int = 32, learning_rate: float = 5e-5,
                weight_decay: float = 0.01, max_seq_length: int = 128, do_lower_case: bool = True,
                do_test: bool = True):
    training_parameters = {'model_name': model_name, 'epochs': epochs,
                           'output_size': output_size, 'batch_size': batch_size, 'learning_rate': learning_rate,
                           'weight_decay': weight_decay, 'max_seq_length': max_seq_length,
                           'do_lower_case': do_lower_case}

    output_dir = manage_output_dir(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=output_size,
        cache_dir=os.path.join(MODELS_FOLDER, model_name)
    )
    logger.info(f"Model {model_name} loaded.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.path.join(MODELS_FOLDER, model_name),
        do_lower_case=do_lower_case
    )
    logger.info(f"Tokenizer {model_name} loaded.")

    logger.info(f"Loading datasets")
    train_dataset = TextDatasetForTransformers(
        filepath=os.path.join(data_dir, 'hotels.sentence.train.txt'),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )
    logger.info("Train dataset loaded.")
    dev_dataset = TextDatasetForTransformers(
        filepath=os.path.join(data_dir, 'hotels.sentence.dev.txt'),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )
    logger.info("Dev dataset loaded.")

    # Define Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,  # strength of weight decay
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )
    # Train pre-trained model
    logger.info("***** Running training *****")
    trainer.train()
    logger.info("Training finished.")

    trainer.save_model(output_dir)
    trainer.save_state()

    output_training_params_file = os.path.join(output_dir, "hp.json")
    dictionary_to_json(training_parameters, output_training_params_file)

    if do_test:
        logger.info(f"Loading test datasets")
        test_dataset = TextDatasetForTransformers(
            filepath=os.path.join(data_dir, 'hotels.sentence.test.txt'),
            tokenizer=tokenizer,
            max_seq_length=max_seq_length
        )
        logger.info("Test dataset loaded.")

        prediction_output = trainer.predict(test_dataset)
        y_true = prediction_output.label_ids
        y_pred = np.argmax(prediction_output.predictions, axis=-1)

        print('\n\t**** Classification report ****\n')
        classes_names = test_dataset.label_encoder.classes_
        print(classification_report(y_true, y_pred, target_names=classes_names))

        report = classification_report(y_true, y_pred, target_names=classes_names, output_dict=True)
        dictionary_to_json(report, os.path.join(output_dir, "test_results.json"))

        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=classes_names, columns=classes_names)
        fig, ax = get_confusion_matrix_plot(df_cm)
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), bbox_inches='tight')


def manage_output_dir(model_name: str) -> str:
    output_dir = os.path.join(MODELS_FOLDER, model_name, 'finetuned')
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"accuracy": accuracy, "f1": f1}
