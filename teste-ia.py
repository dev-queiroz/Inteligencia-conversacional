import os
import pyttsx3
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Inicializa o sintetizador de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Ajusta a velocidade da fala

# Configura o modelo de fala offline (VOSK)
model_path = "models/vosk-model-pt"  # Caminho para o modelo VOSK baixado
if not os.path.exists(model_path):
    raise ValueError("Modelo VOSK não encontrado. Baixe o modelo e ajuste o caminho 'model_path'.")

# Carrega o modelo VOSK
vosk_model = Model(model_path)

# Carrega o modelo GPT-J da EleutherAI
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gptj_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# Define o pipeline para o GPT-J
chatbot = pipeline("text-generation", model=gptj_model, tokenizer=tokenizer, max_length=100)

def ouvir_usuario():
    """Escuta o usuário usando VOSK e converte a fala em texto."""
    print("Diga algo:")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
        recognizer = KaldiRecognizer(vosk_model, 16000)
        audio = []
        while True:
            data = stream.read(4000)[0]
            if recognizer.AcceptWaveform(bytes(data)):
                result = recognizer.Result()
                texto = eval(result)['text']
                print("Você disse:", texto)
                return texto
            else:
                audio.append(np.frombuffer(data, dtype=np.int16))

def gerar_resposta(pergunta):
    """Gera uma resposta baseada na pergunta do usuário usando o modelo GPT-J."""
    resposta = chatbot(
        pergunta,
        max_length=100,
        temperature=0.7,   # Controle de aleatoriedade para respostas mais naturais
        top_k=50,          # Limita a escolha das palavras para uma saída mais coesa
        top_p=0.9          # Considera apenas palavras com probabilidade cumulativa de 90%
    )[0]['generated_text']

    # Remove a pergunta inicial da resposta para deixá-la mais natural
    resposta = resposta[len(pergunta):].strip()
    print("Resposta:", resposta)
    return resposta

def falar_resposta(resposta):
    """Converte a resposta em fala e fala para o usuário."""
    engine.say(resposta)
    engine.runAndWait()

def main():
    while True:
        pergunta = ouvir_usuario()
        if pergunta:
            resposta = gerar_resposta(pergunta)
            falar_resposta(resposta)

if __name__ == "__main__":
    main()
