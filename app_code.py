"""
Aplicación principal para el taller de IA emocional
Este script implementa la interfaz de usuario con Gradio y la lógica
para el clasificador de emociones en texto con sistema de feedback
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# Importar módulos propios
from preprocessing import preprocess_text
from model import train_model, evaluate_model, predict_emotion

# Constantes
EMOTIONS = ["alegría", "tristeza", "enojo"]
DATA_PATH = "data/emotions_subset.csv"
FEEDBACK_PATH = "data/feedback.csv"
MODEL_PATH = "models/classifier_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Crear directorios si no existen
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Variables globales para mantener estado
feedback_data = []
model_metrics = {"accuracy": 0, "f1": 0, "errors": 0}
model = None
vectorizer = None

def load_or_train_model():
    """
    Carga el modelo si existe, o entrena uno nuevo si no
    """
    global model, vectorizer
    
    # Verificar si los archivos del modelo existen
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("Cargando modelo existente...")
        model = pickle.load(open(MODEL_PATH, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    else:
        print("Entrenando nuevo modelo...")
        # Cargar datos
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"No se encontró el archivo de datos: {DATA_PATH}")
        
        data = pd.read_csv(DATA_PATH)
        
        # Preprocesar textos
        processed_texts = [preprocess_text(text) for text in data['text']]
        labels = data['emotion']
        
        # Entrenar modelo
        model, vectorizer = train_model(processed_texts, labels)
        
        # Guardar modelo
        pickle.dump(model, open(MODEL_PATH, 'wb'))
        pickle.dump(vectorizer, open(VECTORIZER_PATH, 'wb'))
        
        # Evaluar y guardar métricas
        metrics = evaluate_model(model, vectorizer, processed_texts, labels)
        model_metrics.update(metrics)
    
    return model, vectorizer

def predict(text):
    """
    Predice la emoción para un texto dado
    """
    global model, vectorizer
    
    if model is None or vectorizer is None:
        model, vectorizer = load_or_train_model()
    
    emotion, probabilities = predict_emotion(text, model, vectorizer, preprocess_text)
    
    # Crear datos para el gráfico de barras
    confidence_data = {
        "emotion": list(EMOTIONS),
        "confidence": [float(p) for p in probabilities]
    }
    
    # Determinar la emoción con mayor probabilidad
    max_emotion = EMOTIONS[np.argmax(probabilities)]
    
    return max_emotion, confidence_data

def save_feedback(text, predicted_emotion, correct_emotion):
    """
    Guarda el feedback del usuario
    """
    global feedback_data
    
    # Crear entrada de feedback
    feedback_entry = {
        "text": text,
        "predicted": predicted_emotion,
        "correct": correct_emotion,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    feedback_data.append(feedback_entry)
    
    # Guardar en CSV
    df = pd.DataFrame(feedback_data)
    df.to_csv(FEEDBACK_PATH, index=False)
    
    return f"Feedback guardado correctamente. Total acumulado: {len(feedback_data)}"

def retrain_with_feedback():
    """
    Reentrenar el modelo incluyendo el feedback del usuario
    """
    global model, vectorizer, model_metrics, feedback_data
    
    # Verificar si hay feedback disponible
    if len(feedback_data) == 0:
        if os.path.exists(FEEDBACK_PATH):
            feedback_df = pd.read_csv(FEEDBACK_PATH)
            if len(feedback_df) > 0:
                feedback_data = feedback_df.to_dict('records')
            else:
                return "No hay feedback disponible para reentrenar"
        else:
            return "No hay feedback disponible para reentrenar"
    
    # Cargar datos originales
    original_data = pd.read_csv(DATA_PATH)
    
    # Preparar datos de feedback para entrenamiento
    feedback_df = pd.DataFrame(feedback_data)
    feedback_texts = list(feedback_df['text'])
    feedback_labels = list(feedback_df['correct'])
    
    # Combinar datasets
    all_texts = list(original_data['text']) + feedback_texts
    all_labels = list(original_data['emotion']) + feedback_labels
    
    # Preprocesar
    processed_texts = [preprocess_text(text) for text in all_texts]
    
    # Reentrenar modelo
    model, vectorizer = train_model(processed_texts, all_labels)
    
    # Guardar modelo actualizado
    pickle.dump(model, open(MODEL_PATH.replace('.pkl', '_updated.pkl'), 'wb'))
    pickle.dump(vectorizer, open(VECTORIZER_PATH.replace('.pkl', '_updated.pkl'), 'wb'))
    
    # Evaluar y guardar métricas
    new_metrics = evaluate_model(model, vectorizer, processed_texts, all_labels)
    old_metrics = model_metrics.copy()
    model_metrics.update(new_metrics)
    
    # Preparar mensaje de comparación
    comparison = f"""
    Métricas antes del reentrenamiento:
    - Accuracy: {old_metrics['accuracy']:.2f}
    - F1-Score: {old_metrics['f1']:.2f}
    
    Métricas después del reentrenamiento:
    - Accuracy: {new_metrics['accuracy']:.2f}
    - F1-Score: {new_metrics['f1']:.2f}
    
    Feedback incorporado: {len(feedback_data)} ejemplos
    """
    
    return comparison

def get_error_examples():
    """
    Obtiene ejemplos de errores del feedback
    """
    if len(feedback_data) == 0 and os.path.exists(FEEDBACK_PATH):
        feedback_df = pd.read_csv(FEEDBACK_PATH)
        feedback_data.extend(feedback_df.to_dict('records'))
    
    if len(feedback_data) == 0:
        return "No hay ejemplos de error disponibles."
    
    error_examples = [f for f in feedback_data if f['predicted'] != f['correct']]
    
    if len(error_examples) == 0:
        return "No se encontraron ejemplos de error en el feedback."
    
    # Limitar a máximo 5 ejemplos
    examples = error_examples[:5]
    
    result = "### Ejemplos de errores del modelo:\n\n"
    for i, ex in enumerate(examples, 1):
        result += f"**Ejemplo {i}:**\n"
        result += f"- Texto: \"{ex['text']}\"\n"
        result += f"- Predicción errónea: {ex['predicted']}\n"
        result += f"- Corrección humana: {ex['correct']}\n\n"
    
    return result

def create_gradio_interface():
    """
    Crear y configurar la interfaz de Gradio
    """
    with gr.Blocks(title="Detector de Emociones en Texto") as demo:
        gr.Markdown("# 🧠 IA que detecta emociones en texto... y se equivoca")
        gr.Markdown("""
        Escribe un texto y el modelo intentará detectar la emoción principal (alegría, tristeza o enojo).
        Luego podrás corregir al modelo si se equivoca y reentrenarlo para que mejore.
        """)
        
        with gr.Tab("Detector de emociones"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Escribe un texto para analizar",
                        placeholder="Ej: Hoy recibí una gran noticia que me hizo saltar de felicidad",
                        lines=3
                    )
                    analyze_btn = gr.Button("Analizar emoción", variant="primary")
                
                with gr.Column(scale=2):
                    emotion_output = gr.Textbox(label="Emoción detectada")
                    confidence_plot = gr.BarPlot(
                        label="Confianza por cada emoción",
                        x="emotion",
                        y="confidence",
                        title="Distribución de probabilidades",
                        tooltip=["emotion", "confidence"],
                        y_lim=[0, 1],
                        color="emotion"
                    )
            
            with gr.Row():
                gr.Markdown("### ¿El modelo se equivocó? Envía tu corrección:")
                
            with gr.Row():
                with gr.Column():
                    correct_emotion = gr.Radio(
                        EMOTIONS,
                        label="Selecciona la emoción correcta"
                    )
                    feedback_btn = gr.Button("Enviar corrección")
                
                with gr.Column():
                    feedback_output = gr.Textbox(
                        label="Estado del feedback",
                        lines=2
                    )
        
        with gr.Tab("Reentrenamiento y análisis"):
            with gr.Row():
                retrain_btn = gr.Button("Reentrenar con feedback", variant="primary")
                metrics_output = gr.Textbox(
                    label="Comparación de métricas",
                    lines=8
                )
            
            with gr.Row():
                errors_btn = gr.Button("Ver ejemplos de errores")
                errors_output = gr.Markdown(
                    label="Ejemplos de errores del modelo"
                )
        
        # Configurar eventos
        analyze_btn.click(
            predict,
            inputs=text_input,
            outputs=[emotion_output, confidence_plot]
        )
        
        feedback_btn.click(
            save_feedback,
            inputs=[text_input, emotion_output, correct_emotion],
            outputs=feedback_output
        )
        
        retrain_btn.click(
            retrain_with_feedback,
            inputs=[],
            outputs=metrics_output
        )
        
        errors_btn.click(
            get_error_examples,
            inputs=[],
            outputs=errors_output
        )
    
    return demo

if __name__ == "__main__":
    # Cargar o entrenar modelo
    load_or_train_model()
    
    # Crear y lanzar interfaz
    demo = create_gradio_interface()
    demo.launch(share=True)
    print("Aplicación iniciada en http://localhost:7860")
