import os
import pandas as pd
import re
import json

def pre_process_data(df, is_test=False):
    df = df.fillna("")
    df = df.astype(str)
    df = df.applymap(lambda x: x.lower())
    # Retira pontuações. Útil para ganhar acurácia, O LLM as vezes retorna a frase corretamente, mas sem ponto final. 
    # Logo, pontuação no nosso caso é irrelevante.
    df = df.applymap(lambda x: re.sub(r'[^\w\s]', '', x))
    # Embaralha o conjunto de teste
    if is_test: 
        df = df.sample(frac=1).reset_index(drop=True)
    # Se não for o conjunto de teste, é o output. Remove as colunas start_time e end_time (não é necessário fazer isso no test_set, no arquivo já não há as colunas)
    else: 
        df =  df.drop(columns=['start_time', 'end_time'])
    return df


def save_test_shuffled(test_set):
    test_json = test_set.to_json(orient="records", indent=4)
    with open("eval-data/test_recente.json", "w") as f:
        f.write(test_json)

def save_results(file, results):
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file, 'w') as f:
        json.dump(results, f, indent=4)


def get_test_set(test_path):
    test_set =  pd.read_json(test_path)
    test_set = pre_process_data(test_set, True)
    return test_set

