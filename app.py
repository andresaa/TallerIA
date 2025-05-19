
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


import os

import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('stopwords')

data = {
    'text' : [
        # Alegría
        'Estoy muy feliz con mi nuevo trabajo',
        'Qué alegría ver a mis amigos de nuevo',
        'Me encanta pasar tiempo con mi familia',
        'Estoy tan contento por el resultado',
        'Hoy es el mejor día de mi vida',
        'Acabo de recibir una gran noticia',
        'Celebrando mi cumpleaños con las personas que amo',
        'Por fin terminé mi proyecto y quedó perfecto',
        'Qué felicidad me da verte después de tanto tiempo',
        'Conseguí el trabajo de mis sueños',
        
        # Enojo
        'Odio cuando las cosas no salen como quiero',
        'Estoy furioso por el mal servicio',
        'Me molesta que no me escuchen',
        'No soporto que me mientan en la cara',
        'Estoy harto de que me falten al respeto',
        'Me enfurece la injusticia que estoy viviendo',
        'Qué rabia me da que siempre lleguen tarde',
        'No puedo creer que hayan cancelado a última hora',
        'Me irrita profundamente su actitud prepotente',
        'Estoy cansado de repetir las mismas instrucciones',
        
        # Tristeza
        'Hoy me siento triste y solo',
        'Siento una profunda tristeza hoy',
        'Estoy decepcionado con la situación',
        'Extraño mucho a mi familia que está lejos',
        'Me duele saber que no volveré a verle',
        'Qué pena me da que las cosas terminaran así',
        'No puedo superar esta pérdida tan grande',
        'Me siento vacío desde que te fuiste',
        'Cada día es más difícil sin ti a mi lado',
        'Lloro cada vez que recuerdo lo que pasó',
        
        # Sarcasmo - Alegría (pero realmente expresan otras emociones)
        'Vaya, qué alegría, me han cancelado los planes por quinta vez',
        'Oh genial, otra vez llueve cuando tengo que salir, justo lo que necesitaba',
        'Me encanta cuando el tren se retrasa, es lo mejor del mundo',
        'Qué maravilla, más trabajo para el fin de semana',
        'Estoy feliz de que mi vecino ponga música a todo volumen a las 3 de la mañana',
        'Qué día tan perfecto, primero pierdo las llaves y ahora esto',
        'Me fascina esperar tres horas en la fila para nada',
        'Adoro cuando la gente no responde mis mensajes urgentes',
        'Qué alegría tan grande recibir otra factura sorpresa',
        'Estoy encantado de que todos me cancelen a última hora',
        
        # Sarcasmo - Enojo (pero expresado con aparente calma)
        'No, para nada me molesta que uses mi computadora sin permiso',
        'Claro, llega dos horas tarde, no hay problema',
        'Por supuesto que no me importa hacer tu trabajo también',
        'Qué considerado de tu parte avisarme con 5 minutos de antelación',
        'Me parece perfecto que ignores todo lo que dije ayer',
        'No, no estoy enojado, solo me encanta repetir lo mismo 20 veces',
        'Gracias por tu puntualidad, solo te esperé 40 minutos',
        'Qué detalle tan bonito dejar todos los platos sucios',
        'Me encanta cuando prometes algo y luego te olvidas completamente',
        'No, no me molesta en absoluto que interrumpas cada vez que hablo',
        
        # Sarcasmo - Tristeza (ocultando tristeza con humor)
        'Estoy genial, solo son lágrimas de felicidad',
        'No pasa nada, estoy acostumbrado a que me dejen de lado',
        'Qué importa que no me invitaran, seguro la fiesta estaba aburrida',
        'No necesito su apoyo, siempre he estado solo y me va fenomenal',
        'Claro que no me afecta, solo es el rechazo número 15 esta semana',
        'Estoy perfectamente bien solo en casa un sábado por la noche, otra vez',
        'No, no extraño a nadie, estas fotos antiguas las miro por diversión',
        'Qué bueno que nadie recordó mi cumpleaños, no me gustan las celebraciones',
        'Me encanta pasar las fiestas trabajando mientras todos celebran',
        'Es maravilloso ver cómo todos avanzan en la vida mientras yo sigo igual'
    ],

    'emotion' : [
        # Alegría (10)
        'alegria', 'alegria', 'alegria', 'alegria', 'alegria',
        'alegria', 'alegria', 'alegria', 'alegria', 'alegria',
        
        # Enojo (10)
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        
        # Tristeza (10)
        'tristeza', 'tristeza', 'tristeza', 'tristeza', 'tristeza',
        'tristeza', 'tristeza', 'tristeza', 'tristeza', 'tristeza',
        
        # Sarcasmo - clasificado como enojo (10)
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        
        # Sarcasmo - clasificado como enojo (10)
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        'enojo', 'enojo', 'enojo', 'enojo', 'enojo',
        
        # Sarcasmo - clasificado como tristeza (10)
        'tristeza', 'tristeza', 'tristeza', 'tristeza', 'tristeza',
        'tristeza', 'tristeza', 'tristeza', 'tristeza', 'tristeza'
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

port = int(os.environ.get("PORT", 8080))
demo.launch(server_port=port, server_name="0.0.0.0")
