"""
Módulo de preprocesamiento de texto para el taller de IA emocional.
Contiene funciones para limpieza y transformación de texto.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Descargar recursos de NLTK si no están presentes
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inicializar stemmer para español
stemmer = SnowballStemmer('spanish')

def preprocess_text(text, language='spanish'):
    """
    Aplica preprocesamiento básico a un texto:
    - Convertir a minúsculas
    - Eliminar caracteres especiales
    - Tokenizar
    - Eliminar stopwords
    - Aplicar stemming
    
    Args:
        text (str): Texto a preprocesar
        language (str): Idioma para stopwords y stemming
    
    Returns:
        str: Texto preprocesado
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Eliminar emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Eliminar etiquetas HTML si existieran
    text = re.sub(r'<.*?>', '', text)
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenizar
    tokens = word_tokenize(text, language=language)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Aplicar stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Unir tokens
    processed_text = ' '.join(tokens)
    
    return processed_text

def get_meaningful_words(text, n=10):
    """
    Obtiene las palabras más significativas de un texto
    (excluyendo stopwords)
    
    Args:
        text (str): Texto a analizar
        n (int): Número de palabras a devolver
    
    Returns:
        list: Lista de palabras significativas
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Tokenizar
    tokens = word_tokenize(text, language='spanish')
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    meaningful_words = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Obtener las n palabras más frecuentes
    from collections import Counter
    word_counts = Counter(meaningful_words)
    return [word for word, _ in word_counts.most_common(n)]

def extract_emotion_features(text):
    """
    Extrae características adicionales relacionadas con emociones
    
    Args:
        text (str): Texto a analizar
    
    Returns:
        dict: Diccionario con características
    """
    features = {}
    
    # Longitud del texto
    features['text_length'] = len(text)
    
    # Número de palabras
    features['word_count'] = len(text.split())
    
    # Presencia de signos de exclamación
    features['has_exclamation'] = int('!' in text)
    
    # Presencia de emojis básicos (simplificado)
    features['has_happy_emoji'] = int(any(emoji in text for emoji in [':)', ':-)', ':D', '=)']))
    features['has_sad_emoji'] = int(any(emoji in text for emoji in [':(', ':-(', '=(']))
    
    # Palabras en mayúsculas (potencial indicador de énfasis/emoción)
    words = text.split()
    features['uppercase_ratio'] = sum(1 for w in words if w.isupper()) / len(words) if words else 0
    
    return features

def highlight_emotional_words(text, emotion_dict=None):
    """
    Destaca palabras potencialmente emocionales en un texto
    
    Args:
        text (str): Texto a analizar
        emotion_dict (dict): Diccionario de palabras emocionales (opcional)
    
    Returns:
        str: Texto con palabras destacadas en formato HTML
    """
    if emotion_dict is None:
        # Diccionario simple de ejemplo
        emotion_dict = {
            'alegría': ['feliz', 'contento', 'alegre', 'encantado', 'maravilloso', 'fantástico'],
            'tristeza': ['triste', 'deprimido', 'melancólico', 'fatal', 'horrible', 'desgraciado'],
            'enojo': ['enfadado', 'furioso', 'molesto', 'irritado', 'indignado', 'cabreado']
        }
    
    # Tokenizar el texto
    tokens = word_tokenize(text.lower(), language='spanish')
    
    # Crear mapa de palabras a emociones
    word_to_emotion = {}
    for emotion, words in emotion_dict.items():
        for word in words:
            word_to_emotion[word] = emotion
    
    # Crear versión HTML con palabras destacadas
    html_text = text
    for token in tokens:
        if token in word_to_emotion:
            emotion = word_to_emotion[token]
            color = {
                'alegría': 'green',
                'tristeza': 'blue',
                'enojo': 'red'
            }.get(emotion, 'black')
            
            # Reemplazar token con versión coloreada (case insensitive)
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            replacement = f'<span style="color:{color};font-weight:bold">{token}</span>'
            html_text = pattern.sub(replacement, html_text)
    
    return html_text
