# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import OrdinalEncoder
# import numpy as np
# from fuzzywuzzy import fuzz
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import pandas as pd
import json
from utils import *

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
    output_df = pre_process_data(output_df)
    with open('eval-data/output_recente.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    return output_df

def calcularResultados(tp, fp, fn, tn):
    precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4)
    recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)
    f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 4)
    
    return precision, recall, f1


def calculateMetrics(pred_label, test_label, target_label):
    """
    Calcula métricas precisao, revocação, f1 e acurácia para o label 'category' ou 'action'
    """
    unique_labels = set(pred_label).union(set(test_label))
    unique_labels.discard("")
    results = []
    for label in unique_labels:
        # Condições para cada métrica
        # True Positive: previu a categoria e era a categoria
        tp = float(((pred_label == label) & (test_label == label)).sum())
        # False Positive: previu a categoria e não era
        fp = float(((pred_label == label) & (test_label != label)).sum())
        # False Negative: não previu a categoria, mas de fato era
        fn = float(((pred_label != label) & (test_label == label)).sum())
        # True Negatives: não previu a categoria e de fato não era
        tn = float(((pred_label != label) & (test_label != label)).sum())

        precision, recall, f1 = calcularResultados(tp, fp, fn, tn)
        new_json = {
            target_label: label,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
        }
        results.append(new_json)
    return results




def media_por_label(teste, output, target_label):
    """
    Essa função faz o mesmo que a função "media_por_categoria" usada no repo 'llm_evaluation'
    Porém mais flexível, podendo ser usada para calcular o quanto que, para cada action ou category, o LLM acerta os outros paramêtros
    """
    unique_labels = set(output[target_label]).union(set(teste[target_label]))
    unique_labels.discard("")
    media = {label: 0 for label in unique_labels}
    media['evaluation'] = "avg accuracy of parameters"
    media['label'] = target_label
    total_labels = len(teste.columns)
    for label in unique_labels:
        indices = output[output[target_label] == label].index
        linhas_label_teste = teste.loc[indices]
        linhas_label_output = output.loc[indices]
        for (_, row_test), (_, row_out) in zip(linhas_label_teste.iterrows(), linhas_label_output.iterrows()):
            acertos = sum(val_test == val_out for val_test, val_out in zip(row_test.values, row_out.values))/total_labels
            media[label] += acertos
        media[label] = round(media[label] / len(linhas_label_output) if len(linhas_label_output) > 0 else 0, 4)
    media['label'] = target_label
    return [media]


def evaluation_llms(test_file_path, test_size, model_name, current_model):
    test_set = get_test_set(test_file_path)
    save_test_shuffled(test_set.head(test_size))
    output_set = newReq(test_set.head(test_size), current_model)
    
    metrics_categoria = calculateMetrics(output_set['category'].head(test_size), test_set['category'].head(test_size), 'category')
    media_acertos_categoria = media_por_label(test_set.head(test_size), output_set.head(test_size), 'category')
    metrics_action = calculateMetrics(output_set['action'].head(test_size), test_set['action'].head(test_size), 'action')
    media_acertos_action = media_por_label(test_set.head(test_size), output_set.head(test_size), 'action')
    # Junta as metricas com a media de acerto e salva
    results = [{'intents': test_size}] + metrics_categoria + media_acertos_categoria + metrics_action + media_acertos_action
    file_path = f"./results/{model_name}/{current_model}.json"
    save_results(file_path, results)
    





