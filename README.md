baixa o ollama em https://ollama.com/download

Na linha de comando:

    ollama pull <nome_do_modelo>
  
    ollama create <nome_do_novo_modelo> --file prompt.modelfile
  

Mude para a pasta do código e rode na linha de comando: pip install -r requirements.txt

No código:

    em main.py, troque modelName pelo <nome_do_novo_modelo>.
    se for uma avaliação de um modelo quantizado, chame "evaluation_llms" com quantized = valor da quantização.
    rodar main.py e já estará avaliando


Avalie a versão normal do modelo, e depois a versão quantizada. (Tome cuidado por que alguns modelos já são quantizados)


Tutorial de como quantizar: https://plainenglish.io/blog/lets-quantize-llama3. Tente generalizar isso pro seu modelo


adicione o tipo do modelo usado em evaluated_models.txt para evitar ser avaliado de novo.

