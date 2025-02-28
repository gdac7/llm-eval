# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from tqdm import tqdm
# from langchain_ollama import OllamaLLM
# import pandas as pd
# import json
# from sklearn.metrics import accuracy_score
# from fuzzywuzzy import fuzz
# from new_eval import *
# import os

# def save_test_shuffled(test_set):
#     test_json = test_set.to_json(orient="records", indent=4)
#     with open("eval-data/test_recente.json", "w") as f:
#         f.write(test_json)





# def req(input_df, new_test_set, model): 
#     columns = ["intent", "category", "action", "requirement", "targets", "magnitude", "start_time", "end_time"]
#     output_df = pd.DataFrame(columns=columns)
#     falhas = []
#     json_to_save = []
#     i = 0
#     for _, row  in tqdm(input_df.iterrows(), desc="Obtendo outputs"):
#         llm = OllamaLLM(model=model)
#         intent = row['intent']
#         response = llm.invoke(intent)
#         if response:
#             response = response.strip()
#             #print(response)
#             try:
#                 response = json.loads(response)
#             except json.JSONDecodeError as e:
#                 new_test_set = new_test_set.drop(index=i).reset_index(drop=True)
#                 falhas.append((response, i, e))
#                 #output_df = output_df.drop(index=1)
#                 continue
#             json_to_save.append(response)
#             new_row = pd.DataFrame([response])
#             output_df = pd.concat([output_df, new_row], ignore_index=True)
#         else:
#             print("Nao respondeu nada")
        
        
#         #print(new_test_set)
        
        
# #         i += 1


# #     output_df = output_df.drop(columns=['start_time', 'end_time'])
# #     output_df = output_df.fillna("")
# #     output_df = output_df.astype(str)
# #     output_df = output_df.applymap(lambda x: x.lower())
# #     with open('eval-data/output_recente.json', 'w') as f:
# #         json.dump(json_to_save, f, indent=4)  

   
  
# #     return output_df, new_test_set      



# def metrics_by_category(test, output):
#     # Combinar os DataFrames test e output para comparação
#     merged = test[['category', 'action']].merge(output[['category', 'action']], 
#                                                 left_index=True, 
#                                                 right_index=True, 
#                                                 suffixes=('_true', '_pred'))

#     # Filtrar categorias únicas
#     categories = merged['category_true'].unique()

#     # Calcular métricas para cada categoria
#     metrics_per_category = {}
#     for category in categories:
#         # Filtrar apenas as linhas correspondentes à categoria atual
#         category_data = merged[merged['category_true'] == category]
        
#         # Calcular precisão, recall e F1-Score para essa categoria
#         precision = precision_score(category_data['category_true'], category_data['category_pred'], average='weighted', zero_division=0)
#         recall = recall_score(category_data['category_true'], category_data['category_pred'], average='weighted', zero_division=0)
#         f1 = f1_score(category_data['category_true'], category_data['category_pred'], average='weighted', zero_division=0)
        
#         metrics_per_category[category] = {
#             'precision': f"{precision:.2f}",
#             'recall': f"{recall:.2f}",
#             'f1_score': f"{f1:.2f}"
#         }

#     return metrics_per_category

# def metrics_per_label(test, input) -> dict:
#     metrics_per_label = {}
#     for column in test:
#         precision = precision_score(test[column], input[column], average='weighted', zero_division=0)
#         recall = recall_score(test[column], input[column], average='weighted', zero_division=0)
#         f1 = f1_score(test[column], input[column], average='weighted', zero_division=0)
        
#         metrics_per_label[column] = {
#             'precision': f"{precision:.2f}",
#             'recall': f"{recall:.2f}",
#             'f1_score': f"{f1:.2f}"
#         }
        
#     return metrics_per_label

# def metrics_per_label_with_fuzzy(test, input, threshold=70) -> dict:
#     metrics_per_label = {}

#     for column in test:
#         true_labels = []
#         pred_labels = []

#         for t, i in zip(test[column], input[column]):
#             # Calcular a similaridade fuzzy
#             similarity = fuzz.token_sort_ratio(str(t), str(i))
            
#             # Verificar se a similaridade está acima do limiar
#             if similarity >= threshold:
#                 true_labels.append(1)  # Match correto
#                 pred_labels.append(1)
#             else:
#                 true_labels.append(1)  # Sempre conta como positivo no teste
#                 pred_labels.append(0)  # Match incorreto
        
#         # Calcular precisão, recall e F1-Score para a coluna
#         precision = precision_score(true_labels, pred_labels, zero_division=0)
#         recall = recall_score(true_labels, pred_labels, zero_division=0)
#         f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
#         metrics_per_label[column] = {
#             'precision': f"{precision:.2f}",
#             'recall': f"{recall:.2f}",
#             'f1_score': f"{f1:.2f}"
#         }

#     return metrics_per_label


# # def accuracy_by_category(test, output):
# #     # Combinar os DataFrames test e output para comparação
# #     merged = test[['category', 'action']].merge(output[['category', 'action']], 
# #                                                 left_index=True, 
# #                                                 right_index=True, 
# #                                                 suffixes=('_true', '_pred'))

# #     # Filtrar categorias únicas
# #     categories = merged['category_true'].unique()

# #     # Calcular acurácia para cada categoria
# #     accuracy_per_category = {}
# #     for category in categories:
# #         # Filtrar apenas as linhas correspondentes à categoria atual
# #         category_data = merged[merged['category_true'] == category]
        
# #         # Calcular a acurácia para essa categoria
# #         accuracy = accuracy_score(category_data['category_true'], category_data['category_pred'])
# #         accuracy_per_category[category] = accuracy

# #     return accuracy_per_category



# # def accuracy_per_label_with_fuzzy(test, input, threshold=70) -> dict:
# #     accuracy_per_label = {}
    
# #     for column in test:
# #         matches = []
# #         for t, i in zip(test[column], input[column]):
# #             # Calcular a similaridade fuzzy
# #             similarity = fuzz.token_sort_ratio(str(t), str(i))
            
# #             # Verificar se a similaridade está acima do limiar
# #             if similarity >= threshold:
# #                 matches.append(1)  # Conta como correto
# #             else:
# #                 matches.append(0)  # Conta como incorreto
        
# #         # Calcular a acurácia considerando os matches fuzzy
# #         accuracy_per_label[column] = sum(matches) / len(matches)
    
# #     return accuracy_per_label

# # def accuracy_per_label(test, input) -> dict:
# #     accuracy_per_label = {}
# #     for column in test:
# #         accuracy_per_label[column] = accuracy_score(test[column], input[column])
        
# #     return accuracy_per_label




# def save_results(file, test_size, acc_class, acc_category, acc_fuzzy):
#     directory = os.path.dirname(file)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     with open(file, 'a') as f:
#         f.write(f"##Intenções testadas: {test_size}\n")
#         f.write(f"  Acurácia por classe (label): \n{acc_class}\n\n")
#         f.write(f"  Acurácia por categoria: \n{acc_category}\n\n")
#         f.write(f"  Acurácia com fuzzy: \n{acc_fuzzy}\n\n")


# def get_test_set(test_path):
#     test_set =  pd.read_json(test_path)
#     test_set = test_set.sample(frac=1).reset_index(drop=True)
#     test_set = test_set.fillna("")
#     test_set = test_set.astype(str)
#     return test_set






    

# def evaluation_llms(test_file_path, test_size, modelName):
#     output_set = pd.read_json("eval-data/output_recente.json")
#     output_set = output_set.fillna("").astype(str).applymap(lambda x: x.lower())
#     output_set = output_set.drop(columns=['start_time', 'end_time'])
#     test_set = pd.read_json("./eval-data/test_recente.json")
#     # test_size = 15
#     # test_set = get_test_set(test_file_path)
#     # intents = pd.DataFrame(test_set['intent'])
#     # save_test_shuffled(test_set.head(test_size))
#     # # Para podermos verificar como o modelo se comporta conforme aumentamos os
#     # output_set = newReq(intents.head(test_size), modelName)
#     metrics_cat = calculateMetricsCategory(output_set['category'], test_set['category'])
#     print(metrics_cat)
#     # metrics_cat = calculateMetricsCategory(test_set.head(test_size), output, ["construct", "transfer", "regulate"], "category")
#     # for result in metrics_cat:
#     #     print(result)
#     #     print('\n')    

#     # metrics = calcularMetricas(createConfusionMatrixCategories(test_set, output), ["construct", "transfer", "regulate", "error"])
#     # print(metrics)
#     # Calcula as acurácias
#     # metrics_label = metrics_per_label(test_set, output)
#     # metrics_cat = metrics_by_category(test_set, output)
#     # Avaliar com fuzzy
#     # Categoria e action bater 100%, logo são removido da aproximação com fuzzy
#     # test_fuzzy = new_test_set.drop(columns=['category', 'action'])
#     # out_fuzzy = output.drop(columns=['category', 'action'])
#     # metrics_label_fuzzy = metrics_per_label_with_fuzzy(test_fuzzy, out_fuzzy)
#     #save_results(f"./results/{modelName}/metrics.txt", test_size, metrics_label, metrics_cat, metrics_label_fuzzy)



# def createConfusionMatrixCategories(input_df, output_df):
#     y_true = input_df['category'].fillna("error")
#     y_pred = output_df['category'].fillna("error")

#     classes = ["construct", "transfer", "regulate", "error"]

#     cm = confusion_matrix(y_true, y_pred, labels=classes)

#     ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap="viridis")

#     return cm

# def calcularMetricas(cm, classes):
#     if "error" in classes:
#         error_idx = classes.index("error")
#     else:
#         error_idx = None
    
#     metricas = {classe: {"precisao": 0, "recall": 0, "f1_score": 0} for classe in classes}

#     cm = np.array(cm)

#     for i, classe in enumerate(classes):
#         if classe == "error":
#             continue

#         tp = cm[i, i]
#         fp = cm[:, i].sum() - tp
#         fn = cm[i, :].sum() - tp

#         if error_idx is not None:
#             fp -= cm[error_idx, i]
#             fn -= cm[i, error_idx]

#         precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1_score = (2 * precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

#         metricas[classe]["precisao"] = precisao
#         metricas[classe]["recall"] = recall
#         metricas[classe]["f1_score"] = f1_score
    
    # return metricas