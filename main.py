from eval import evaluation_llms

if __name__ == "__main__":
    test_file_path = "eval-data/test_500.json"
    test_set_size = 5
    # Avalie em pares: Modelo não quantizado vs Modelo quantizado.
    all_models = ["agir-llama3:8b"]
    for modelName in all_models:
        evaluation_llms(test_file_path=test_file_path, test_size=test_set_size, modelName=modelName)
    