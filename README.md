## Básico sobre quantização

A quantização, no contexto do projeto, tem o objetivo de reduzir a precisão dos pesos/valores de ativação
(que geralmente são de alta precisão, como ponto flutuante de 32 ou 16 bits) para precisões mais baixas.
Essa técnica implica em certa redução da precisão, mas também aumenta a velocidade de inferência e diminui os custos
computacionais do modelo significativamente. Na redução de FP32 para INT8, a redução de precisão é pequena
o suficiente para poder ser desconsiderada.

A notação que provavelmente será encontrada é:

- **Q8_0:** Usada para representar versões com pesos e valores de ativação quantizados diretamente, nesse caso, para 8 bits;
- **FP32:** Versão com precisão de 32 bits, utilizada geralmente durante o treinamento do modelo, onde a precisão é prioridade;
- **Q4_K_L ou Q4_K_M:** Técnicas de quantização por camada ou kernel para minimizar o impacto da redução de bits, nesse caso, aplicada para 4 bits.

Nosso objetivo é avaliar o comportamento de LLMs, dadas as suas diferentes versões, em criar um JSON que represente
bem a intenção presente em um dataset criado por nós. Como as versões com ponto flutuante consomem muito espaço, energia e
tempo para realizar inferências, utilizaremos versões quantizadas dos modelos, com 8, 4 ou 2 bits.

Para isso, será necessário: baixar o Ollama para o gerenciamento dos LLMs; baixar os arquivos GGUF do HuggingFace; e compilá-los para serem usados.

## Deixando tudo pronto

Baixe o [Ollama](https://ollama.com/download) e siga os passos para a instalação.

Após isso, acesse o HuggingFace [https://huggingface.co/] e navegue para "Models". Nesta página, pesquise pelo modelo desejado e selecione-o.

Na página do modelo, procure por "Quantizations" e selecione uma versão adequada e que possibilite a instalação do arquivo GGUF, geralmente o
nome do arquivo também está a quantização que foi feita, como mostrado na figura abaixo, a qual mostra alguns arquivos GGUF, como os que deverão
ser baixados para a compilação do modelo.

![image](https://github.com/user-attachments/assets/919bbf9c-d609-408c-98af-bd0fc8c5a89b)


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

