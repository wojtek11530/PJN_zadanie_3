import abc
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


class Singleton(abc.ABCMeta,):
    _instances: Dict[object, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_label_encoder(labels: np.ndarray):
    unique_labels = np.sort(np.unique(labels))
    label_encoder = LabelEncoder().fit(unique_labels)
    return label_encoder


def dictionary_to_json(dictionary: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dictionary, f, indent=2)


def is_folder_empty(folder_name: str):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False


def get_confusion_matrix_plot(conf_matrix: pd.DataFrame) -> Tuple[plt.figure, plt.Axes]:
    fig, ax = plt.subplots()
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                       annot_kws={"fontsize": 18}, square=True, ax=ax)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, fontsize=18)
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, fontsize=18)
    ax.set_ylabel('Rzecziwista klasa', fontsize=18)
    ax.set_xlabel('Predykowana klasa', fontsize=18)
    fig.tight_layout()
    return fig, ax