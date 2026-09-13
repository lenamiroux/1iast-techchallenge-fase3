"""Treina o modelo final e salva a pipeline completa (pré-processamento +
classificador) em disco — reutilizável sem precisar retreinar, e ponto de
entrada único e reproduzível pra "a" pipeline de ML do projeto.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.modeling.pipeline import build_model_pipeline
from src.preprocessing.build_features import build_dataset
from src.preprocessing.pipeline import CATEGORICAL, NUMERIC_COMPLETE, NUMERIC_WITH_MISSING
from src.preprocessing.split import split_by_municipio

FEATURE_COLS = NUMERIC_WITH_MISSING + NUMERIC_COMPLETE + CATEGORICAL
MODEL_PATH = Path("reports/models/random_forest.joblib")
TRAIN_SPLIT_PATH = Path("data/raw/train_split.parquet")
TEST_SPLIT_PATH = Path("data/raw/test_split.parquet")


def main() -> None:
    df = pd.read_parquet("data/raw/features_alunos_ml.parquet")
    dataset = build_dataset(df)
    train, test = split_by_municipio(dataset)

    # Guardamos os splits em disco — a etapa de interpretabilidade precisa
    # exatamente do mesmo conjunto de teste, não de um novo split aleatório.
    train.to_parquet(TRAIN_SPLIT_PATH, index=False)
    test.to_parquet(TEST_SPLIT_PATH, index=False)

    model = build_model_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    )
    model.fit(
        train[FEATURE_COLS],
        train["label_alfabetizado"],
        classifier__sample_weight=train["peso_aluno"],
    )

    y_pred = model.predict(test[FEATURE_COLS])
    print(classification_report(test["label_alfabetizado"], y_pred, target_names=["Não alfabetizado", "Alfabetizado"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()