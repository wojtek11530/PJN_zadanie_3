import json
import os
from typing import Any, Dict

import pandas as pd

from src.settings import DATA_FOLDER, MODELS_FOLDER


def main():
    models_subdirectories = list(os.walk(MODELS_FOLDER))

    data = list()
    for subdirectory, _, _ in models_subdirectories:
        if 'run' in os.path.basename(subdirectory):
            try:
                data_dict = gather_results(subdirectory)
                data.append(data_dict)
            except Exception as e:
                print(e)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results_lista_3_mlp.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, 'hp.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = test_data_dict

    if 'model_name' in training_data_dict:
        data['full_name'] = training_data_dict['model_name'] + '_' + os.path.basename(model_dir)
        data['name'] = training_data_dict['model_name']
    else:
        data['full_name'] = os.path.basename(model_dir)
        data['name'] = data['full_name'].split('-run-')[0]

    return data


if __name__ == '__main__':
    main()
