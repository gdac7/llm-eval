from langchain_ollama import OllamaLLM
import time

llm = OllamaLLM("llama3:8b:latest")

entrada = input("Digite o seu comando... ").lower()
start_time = time.time()
response = llm.invoke(entrada)
end_time = time.time()

print(response)
print(f"Latencia: {start_time - end_time}")