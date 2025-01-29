from eval import evaluation_llms

if __name__ == "__main__":
    test_file_path = "eval-data/test_500.json"
    test_set_size = 500
    # Nome do modelo que irá ser avaliado (Independente da quantização)
    modelBase = ["llama", "phi"]
    # Avalie em pares: Modelo não quantizado vs Modelo quantizado.
    modelVersions = ["llama3.1-q4km:latest", "phi4-q4km:latest"]
    # Ex: modelBase = ["llama", "llama", "phi"]
    #     modelVersions = ['llama3:8b', 'llama3:16b', 'phi8b']
    # A iteração será feita em (llama, llama3:8b), (llama, llama3:16b), (phi, phi8b).
    for modelName, modelVariant in zip(modelBase, modelVersions):
        evaluation_llms(test_file_path=test_file_path, test_size=test_set_size, model_name=modelName, current_model=modelVariant)
    