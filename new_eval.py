from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score
from fuzzywuzzy import fuzz
import os

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

    with open('eval-data/output_recente.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    return output_df

def createConfusionMatrixCategories(input_df, output_df, label, classes):
    y_true = input_df[label].fillna("error")
    y_pred = output_df[label].fillna("error")

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap="viridis")

    return cm

def calcularMetricas(cm, classes):
    if "error" in classes:
        error_idx = classes.index("error")
    else:
        error_idx = None
    
    metricas = {classe: {"precisao": 0, "recall": 0, "f1_score": 0} for classe in classes}

    cm = np.array(cm)

    for i, classe in enumerate(classes):
        if classe == "error":
            continue

        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        if error_idx is not None:
            fp -= cm[error_idx, i]
            fn -= cm[i, error_idx]

        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

        metricas[classe]["precisao"] = precisao
        metricas[classe]["recall"] = recall
        metricas[classe]["f1_score"] = f1_score
    
    return metricas

def getPositivesNegatives(test_df, output_df, label, classes):
    output = output_df[label]
    correct = test_df[label]

    true_positives = [0]*len(classes)
    false_positives = [0]*len(classes)
    false_negatives = [0]*len(classes)

    for (idx1, row1), (idx2, row2) in zip(correct.iterrows(), output.iterrows()):
        if row1[label] == row2[label]:
            true_positives[classes.index(row2[label])] += 1
        if row2[label] != "error" and row1[label] != row2[label]:
            false_positives[classes.index(row2[label])] += 1
            false_negatives[classes.index(row1[label])] += 1

    return true_positives, false_positives, false_negatives