from eval import evaluation_llms

if __name__ == "__main__":
    test_file_path = "eval-data/test_500.json"
    test_set_size = 10
    # Se quiser testar varios modelos de uma só vez, bota o modelo na lista da forma de tuple (nome_do_modelo, quantização) -> None para não quantizado 
    all_models = [("customLlama:latest", None)]
    for modelName, quantization in all_models:
        evaluation_llms(test_file_path=test_file_path, test_size=test_set_size, 
                        modelName=modelName, quantized=quantization)
    