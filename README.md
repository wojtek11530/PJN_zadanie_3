# Przetwarzanie Języka Naturalnego - zadanie 3

Modele klasyfikujące napisany z wykorzystaniem bibliotek PyTorch Lightning i HuggingFace.
Dostępne modele:
 - MLP przyjmujący na wejściu średnią embeddingów słów w tekście (embeddinig fastText lub word2vec),
 - model Transformer do klasyfikacji zdań dostępny na HuggingFace.

# Docker

Plik `Dockerfile` zawiera obraz dockerowy umożliwiający uruchomienie projektu z wykorzystaniem GPU (CUDA). 

Budowanie obrazu:
```
docker build -t pjn --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```

Uruchomienie:
```
docker run -it --gpus all --name pjn -v "$(pwd)":/app pjn bash
```

# Model Transformer

Trenowany poprzez skrypt `run_transformer_experiments.py`:
```
python3 -m src.scripts.run_transformer_experiments
```
w którym podaje się hiperparametry, w tym nazwę modelu Transformer dostępnego w repozytorium HuggingFace. 
W wyniku trenowania modelu w folderze `models/<nazwa_modelu>/finetuned` zapisze się checkpoint modelu,
hiperparametry w pliku `hp.json`.

Dodanie flagi `--do_test` dodatkowo uruchomi ewaluacje wytrenowanego modelu na zbiorze testowym, wyniki zapiszą się 
w folderze `models/<nazwa_modelu>/finetuned`.

# Model MLP

## Trenowanie 

Trenowanie modelu uruchamiamy skryptem `train_model`:
```
python -m src.scripts.train_model \
  --data_dir ${MULTIEMO_HOTEL_DATA_DIR} \
  --word_embedding_model_dir ${WORD_EMB_MODEL_PATH} \
  --word_embedding_type ${WORD_EMB_TYPE} \
  --input_size ${INPUT_SIZE} \
  --hidden_size ${HID_SIZE} \
  --output_size ${OUT_SIZE} \
  --dropout ${DROPOUT} \
  --batch_size ${BS} \
  --learning_rate ${LR} \
  --weight_decay ${WD} \
  --eval
```
gdzie wymagane jest by podać:
 - `data_dir` - lokalizacja, gdzie mieszczą się pliki `hotels.sentence.train.pl.txt`, 
   `hotels.sentence.dev.pl.txt`, `hotels.sentence.test.pl.txt`,
 - `word_embedding_model_dir` - ścieżka do pliku modelu word embedding
 - `word_embedding_type` - typ modelu word embeddingu, możliwe wartości to `fasttext` lub `word2vec,
 
Pozostałe argumenty nie są wymagane, gdyż mogą przyjąć wartości domyślne:
 - `input_size` - wymiar wejściowy modelu, równy wymiarowi word embeddingu, domyślnie: 100,
 - `hidden_size` - rozmiar warstwy ukrytej, domyślnie: 32,
 - `output_size` - rozmiar wyjścia równy liczbie klasy, domyślnie: 4,
 - `dropout` - prawdopodobieństwo dropoutu, domyślnie: 0.01,
 - `batch_size` - rozmiar batcha podczas trenowania, domyślnie: 32,
 - `learning_rate` - współczynnik uczenia, domyślnie: 0.005,
 - `weight_decay` - współczynnik _weight decay_, domyślnie: 0.0001,
 - `eval` - opcjonalnie, jeśli chcemy od razu dokonać ewaluacji modelu

Przykładowe wywołanie
```
python -m src.scripts.train_model \
  --data_dir data/multiemo2 \
  --word_embedding_model_dir data/fastext/model_dim100.bin \
  --word_embedding_type fasttext \
  --eval
```

W wyniku trenowania modelu w folderze `models` zapisze się checkpoint modelu, hiperparametry w pliku
`hp.json` oraz logi tensorboarda z trenowania.

## Ewaluacja

Zapisany model można ewaluować przy pomocy skryptu:
```
python -m src.scripts.test_model \
  --model_dir ${MODEL_DIR}
  --data_dir ${MULTIEMO_HOTEL_DATA_DIR}
```
gdzie wymagane jest by podać:
 - `data_dir` - lokalizacja, gdzie mieszczą się pliki `hotels.sentence.train.pl.txt`, 
   `hotels.sentence.dev.pl.txt`, `hotels.sentence.test.pl.txt`; do ewaluacji zostanie wykorzystany zbiór
    `hotels.sentence.test.pl.txt`,
 - `model_dir` - lokalizacja, gdzie znajdują się pliki `model.chkpt` and `hp.json` dla wytrenowanego modelu.

Przykładowe wywołanie
```
python -m src.scripts.test_model \
  --model_dir models/MLP-fasttext-run-1 \
  --data_dir data/multiemo2
```
W wyniku ewaluacji w lokalizacji modelu zapisze się plik `test_results.json` z miarami jakości
oraz `confusion_matrix.pdf` z obrazkiem macierzy pomyłek.

Przy trenowaniu ustawienie falgi `eval` podowuje automatyczną ewaluację na zbiorze testowym, 
więc nie trzeba wywoływać skryptu ewaluacyjnego jeszcze raz.

