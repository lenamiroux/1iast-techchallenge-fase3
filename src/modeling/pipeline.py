"""Combina o pré-processamento (Etapa 6) com um classificador num único
objeto Pipeline do scikit-learn — a "integração do pré-processamento
diretamente ao modelo" que o desafio pede. Isso significa que `.fit()` e
`.predict()` sempre aplicam a mesma transformação, sem risco de esquecer
um passo ou aplicá-lo de forma diferente entre treino e produção.
"""

from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import build_preprocessor


def build_model_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])