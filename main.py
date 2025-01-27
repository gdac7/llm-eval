from eval import evaluation_llms

if __name__ == "__main__":
    test_file_path = "eval-data/test_500.json"
    test_set_size = 500
    # Nome do modelo que irá ser avaliado (Independente da quantização)
    modelName = "llama"
    # Avalie em pares: Modelo não quantizado vs Modelo quantizado.
    all_models = ["llama3:8bQ4_0"]
    # Ex: modelName: Llama
    # all_models = ['llama3:8b', 'llama3:16b']
    for modelVariant in all_models:
        evaluation_llms(test_file_path=test_file_path, test_size=test_set_size, model_name=modelName, current_model=modelVariant)
    