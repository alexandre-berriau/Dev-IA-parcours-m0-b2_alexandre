"""Chargement du modèle Hugging Face et inférence."""

from transformers import pipeline


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache).
    
    """
    return pipeline("text-classification", model=model_name, top_k=None)


def predict_sentiment(classifier, text: str) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis.
      
    """
    raise NotImplementedError("predict_sentiment n'est pas encore implémentée")
