
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('stopwords')
data={
    'text': [

        'Estoy muy feliz con mi nuevo trabajo',

        'Odio cuando las cosas no salen como quiero',

        'Hoy me siento triste y solo',

        'Qué alegría ver a mis amigos de nuevo',

        'Estoy furioso por el mal servicio',

        'Siento una profunda tristeza hoy',

        'Me encanta pasar tiempo con mi familia',

        'Me molesta que no me escuchen',

        'Estoy tan contento por el resultado',

        'Estoy decepcionado con la situación'

    ],

    'emotion': [

        'alegria', 'enojo', 'tristeza', 'alegria', 'enojo',

        'tristeza', 'alegria', 'enojo', 'alegria', 'tristeza'

    ]
}
df = pd.DataFrame(data)

stop_words = set(stopwords.words('spanish'))

def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(preprocess)

X = df['text_clean']
y = df['emotion']
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vec, y)

feedback_data = []

def predict_emotion(text):
    text_clean = preprocess(text)
    vec = vectorizer.transform([text_clean])
    prediction = model.predict(vec)[0]
    return prediction

def feedback(texto, emocion_real):
    feedback_data.append((texto, emocion_real))
    df_feedback = pd.DataFrame(feedback_data, columns=['text', 'emotion'])
    df_feedback['text_clean'] = df_feedback['text'].apply(preprocess)
    df_all = pd.concat([df, df_feedback], ignore_index=True)
    X = df_all['text_clean']
    y = df_all['emotion']
    global model, vectorizer
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    return "Gracias por tu retroalimentación. El modelo ha sido actualizado."

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 IA que detecta emociones… y se equivoca")
    with gr.Row():
        input_text = gr.Textbox(label="Escribe una frase")
        output_emotion = gr.Textbox(label="Emoción detectada")
    btn_predict = gr.Button("Detectar emoción")
    btn_predict.click(fn=predict_emotion, inputs=input_text, outputs=output_emotion)

    gr.Markdown("### ¿Se equivocó la IA? Corrígela abajo:")
    corrected_emotion = gr.Radio(choices=['alegria', 'enojo', 'tristeza'], label="¿Cuál era la emoción correcta?")
    btn_feedback = gr.Button("Enviar retroalimentación")
    feedback_output = gr.Textbox(label="Estado")
    btn_feedback.click(fn=feedback, inputs=[input_text, corrected_emotion], outputs=feedback_output)

demo.launch(server_port=8080, share=True)
