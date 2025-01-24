# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import OrdinalEncoder
# import numpy as np
# from fuzzywuzzy import fuzz
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import pandas as pd
import json
import os

def save_test_shuffled(test_set):
    test_json = test_set.to_json(orient="records", indent=4)
    with open("eval-data/test_recente.json", "w") as f:
        f.write(test_json)

def save_results(file, test_size, acc_class, acc_category, acc_fuzzy):
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file, 'a') as f:
        f.write(f"##Intenções testadas: {test_size}\n")
        f.write(f"  Acurácia por classe (label): \n{acc_class}\n\n")
        f.write(f"  Acurácia por categoria: \n{acc_category}\n\n")
        f.write(f"  Acurácia com fuzzy: \n{acc_fuzzy}\n\n")


def get_test_set(test_path):
    test_set =  pd.read_json(test_path)
    test_set = test_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.fillna("")
    test_set = test_set.astype(str)
    return test_set


def newReq(input_df, model):
    columns = ["intent", "category", "action", "requirement",
                "targets", "magnitude", "start_time", "end_time"]
    output_data = []
    llm = OllamaLLM(model=model)

    for _, row in tqdm(input_df.iterrows(), desc="Obtendo outputs"):
        intent = row['intent']
        response = llm.invoke(intent)

        if response:
            response = response.strip()
            try:
                response_json = json.loads(response)
                output_data.append(response_json)
            except json.JSONDecodeError as e:
                output_data.append({col: "" for col in columns})
                output_data[-1]['category'] = "error"
        else:
            output_data.append({col: "" for col in columns})
            output_data[-1]['category'] = "error"

    output_df = pd.DataFrame(output_data)
    output_df = output_df.fillna("").astype(str).applymap(lambda x: x.lower())
    output_df = output_df.drop(columns=['start_time', 'end_time'])

    with open('eval-data/output_recente.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    return output_df





def calcularResultados(tp, fp, fn):
    try:
        precision = (tp) / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = (tp) / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2*((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1


def calculateMetricsCategory(pred_categorias, test_categorias):
    unique_categories = set(pred_categorias).union(set(test_categorias))
    results = []
    for category in unique_categories:
        # Condições para cada métrica
        tp = float(((pred_categorias == category) & (test_categorias == category)).sum())
        fp = float(((pred_categorias == category) & (test_categorias != category)).sum())
        fn = float(((pred_categorias != category) & (test_categorias == category)).sum())
        precision, recall, f1 = calcularResultados(tp, fp, fn)        
        new_json = {
            'category': category,
            'precision': f"{precision:.2f}",
            'recall': f"{recall:.2f}",
            'f1-score': f"{f1:.2f}",
        }
        results.append(new_json)

    return results


def calculateMetricsAction(test_set, output_set):
    pass

def evaluation_llms(test_file_path, test_size, modelName):
    # Usar os arquivos salvos para testar
    # output_set = pd.read_json("eval-data/output_recente.json")
    # output_set = output_set.fillna("").astype(str).applymap(lambda x: x.lower())
    # output_set = output_set.drop(columns=['start_time', 'end_time'])
    # test_set = pd.read_json("./eval-data/test_recente.json")
    test_size = 15
    test_set = get_test_set(test_file_path)
    intents = pd.DataFrame(test_set['intent'])
    save_test_shuffled(test_set.head(test_size))
    # Para podermos verificar como o modelo se comporta conforme aumentamos os
    output_set = newReq(intents.head(test_size), modelName)
    metrics_cat = calculateMetricsCategory(output_set['category'], test_set['category'])
    print(metrics_cat)





