import speech_recognition as sr
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import string
import serial
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# === Paths ===
MODEL_PATH = "C:/Users/ADMIN/Documents/Semester 8/NLP/PBL/try1/bert_emotion_model.pth"

# === Emotion Mapping ===
index_to_emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# === Load Tokenizer & Model ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# === Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# === Speech Recognition ===
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"📝 Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ Request error: {e}")
        return None

# === Emotion Prediction ===
def predict_emotion(text):
    if not text:
        return "unknown"

    clean_text = preprocess_text(text)
    encoding = tokenizer(
        clean_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    prediction = probabilities.detach().cpu().numpy()
    print("Raw model predictions:", prediction)
    print("Predicted index:", predicted_class)

    emotion = index_to_emotion[predicted_class]
    return emotion

# === Send Emotion to ESP32 ===
def send_to_esp32(emotion, port='COM3', baudrate=115200):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            time.sleep(2)  # Give ESP32 time to reboot if needed
            ser.write((emotion + '\n').encode('utf-8'))
            print(f"📤 Sent to ESP32: {emotion}")
    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")

# === Load LLM for contextual response ===
llm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
llm_model = GPT2LMHeadModel.from_pretrained('gpt2')
llm_model.eval()

# === Generate response based on emotion ===
def generate_llm_response(emotion):
    prompt = f"The person is feeling {emotion}. Respond supportively in one sentence:"
    inputs = llm_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = llm_model.generate(
            inputs,
            max_length=40,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    generated_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.replace(prompt, "").strip().split(".")[0] + "."
    print(f"💬 LLM Response: {response}")
    return response

# === Main ===
if __name__ == "__main__":
    spoken_text = recognize_speech()
    emotion = predict_emotion(spoken_text)
    print(f"😃 Detected Emotion: {emotion}")
    send_to_esp32(emotion, port='COM3')  # Update COM port if needed

    # === LLM-based response ===
    llm_response = generate_llm_response(emotion)
