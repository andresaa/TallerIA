"""
Módulo del modelo de clasificación para el taller de IA emocional.
Contiene funciones para entrenar, evaluar y usar el clasificador de emociones.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from preprocessing import preprocess_text

def train_model(texts, labels, test_size=0.2, random_state=42):
    """
    Entrena un modelo de clasificación de emociones
    
    Args:
        texts (list): Lista de textos procesados
        labels (list): Lista de etiquetas correspondientes
        test_size (float): Proporción del dataset para pruebas
        random_state (int): Semilla aleatoria para reproducibilidad
    
    Returns:
        tuple: (modelo entrenado, vectorizador)
    """
    # Vectorizar textos
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.85)
    X = vectorizer.fit_transform(texts)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Entrenar modelo
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, texts, labels, test_size=0.2, random_state=42):
    """
    Evalúa un modelo de clasificación de emociones
    
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        texts (list): Lista de textos preprocesados
        labels (list): Lista de etiquetas correspondientes
        test_size (float): Proporción del dataset para pruebas
        random_state (int): Semilla aleatoria para reproducibilidad
    
    Returns:
        dict: Métricas de evaluación
    """
    # Vectorizar textos
    X = vectorizer.transform(texts)
    
    # Dividir en entrenamiento y prueba
    _, X_test, _, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Predecir etiquetas
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Contar errores
    errors = sum(1 for true, pred in zip(y_test, y_pred) if true != pred)
    
    # Imprimir reporte de clasificación
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(cm)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "errors": errors,
        "confusion_matrix": cm.tolist()
    }

def predict_emotion(text, model, vectorizer, preprocess_fn=None):
    """
    Predice la emoción para un texto dado
    
    Args:
        text (str): Texto a clasificar
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        preprocess_fn (function): Función de preprocesamiento
    
    Returns:
        tuple: (emoción predicha, probabilidades)
    """
    # Preprocesar texto si se proporciona función
    if preprocess_fn:
        processed_text = preprocess_fn(text)
    else:
        processed_text = text
    
    # Vectorizar
    X = vectorizer.transform([processed_text])
    
    # Predecir
    emotion = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return emotion, probabilities

def get_most_important_features(model, vectorizer, class_labels=None, top_n=10):
    """
    Obtiene las características más importantes para cada clase
    
    Args:
        model: Modelo Naive Bayes entrenado
        vectorizer: Vectorizador TF-IDF
        class_labels (list): Etiquetas de clase (opcional)
        top_n (int): Número de características a devolver
    
    Returns:
        dict: Diccionario de características importantes por clase
    """
    feature_names = vectorizer.get_feature_names_out()
    
    if class_labels is None:
        class_labels = model.classes_
    
    # Para modelos Naive Bayes, los coeficientes son log probabilidades
    coefs = model.feature_log_prob_
    
    important_features = {}
    for i, label in enumerate(class_labels):
        # Obtener índices de las características más importantes
        top_indices = coefs[i].argsort()[-top_n:][::-1]
        
        # Obtener nombres de características y sus coeficientes
        top_features = [(feature_names[j], coefs[i][j]) for j in top_indices]
        
        important_features[label] = top_features
    
    return important_features

def analyze_errors(model, vectorizer, texts, true_labels):
    """
    Analiza los errores del modelo
    
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        texts (list): Lista de textos
        true_labels (list): Etiquetas verdaderas
    
    Returns:
        list: Lista de errores (texto, verdadero, predicho)
    """
    # Vectorizar textos
    X = vectorizer.transform(texts)
    
    # Predecir
    predicted_labels = model.predict(X)
    
    # Identificar errores
    errors = []
    for i, (text, true, pred) in enumerate(zip(texts, true_labels, predicted_labels)):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": true,
                "predicted": pred,
                "index": i
            })
    
    return errors

def generate_confusion_examples(model, vectorizer, texts, true_labels, class_labels):
    """
    Genera ejemplos para cada combinación de confusión
    
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        texts (list): Lista de textos
        true_labels (list): Etiquetas verdaderas
        class_labels (list): Lista de etiquetas de clase
    
    Returns:
        dict: Ejemplos por par de confusión
    """
    # Vectorizar textos
    X = vectorizer.transform(texts)
    
    # Predecir
    predicted_labels = model.predict(X)
    
    # Crear diccionario de confusiones
    confusion_examples = {}
    for true_class in class_labels:
        for pred_class in class_labels:
            if true_class != pred_class:
                key = f"{true_class}->{pred_class}"
                confusion_examples[key] = []
    
    # Identificar ejemplos para cada confusión
    for i, (text, true, pred) in enumerate(zip(texts, true_labels, predicted_labels)):
        if true != pred:
            key = f"{true}->{pred}"
            if len(confusion_examples[key]) < 3:  # Limitar a 3 ejemplos por tipo de confusión
                confusion_examples[key].append({
                    "text": text,
                    "true_label": true,
                    "predicted": pred,
                    "index": i
                })
    
    return confusion_examples

def incorporate_feedback(original_texts, original_labels, feedback_texts, feedback_labels):
    """
    Incorpora feedback al conjunto de datos de entrenamiento
    
    Args:
        original_texts (list): Textos originales
        original_labels (list): Etiquetas originales
        feedback_texts (list): Textos de feedback
        feedback_labels (list): Etiquetas de feedback
    
    Returns:
        tuple: (textos combinados, etiquetas combinadas)
    """
    # Combinar listas
    combined_texts = original_texts + feedback_texts
    combined_labels = original_labels + feedback_labels
    
    return combined_texts, combined_labels
