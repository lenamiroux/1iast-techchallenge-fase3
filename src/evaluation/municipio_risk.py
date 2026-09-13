"""Aplica o modelo treinado pra gerar um ranking de municípios por risco
educacional — a "inteligência aplicada à tomada de decisão" que o desafio
pede, não só métricas de modelo. Usa o conjunto de TESTE (fora da amostra
de treino) pra que o ranking reflita capacidade real de generalização,
não memorização.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.pipeline import CATEGORICAL, NUMERIC_COMPLETE, NUMERIC_WITH_MISSING

FEATURE_COLS = NUMERIC_WITH_MISSING + NUMERIC_COMPLETE + CATEGORICAL
MODEL_PATH = Path("reports/models/random_forest.joblib")
TEST_SPLIT_PATH = Path("data/raw/test_split.parquet")
OUTPUT_PATH = Path("reports/municipio_risk_ranking.csv")


def build_ranking() -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    test = pd.read_parquet(TEST_SPLIT_PATH)

    # Probabilidade prevista de o aluno estar alfabetizado (classe 1)
    test = test.copy()
    test["prob_alfabetizado"] = model.predict_proba(test[FEATURE_COLS])[:, 1]

    por_municipio = (
        test.groupby(["ano", "id_municipio"])
        .agg(
            n_alunos_teste=("prob_alfabetizado", "size"),
            prob_media_prevista=("prob_alfabetizado", "mean"),
            taxa_observada=("label_alfabetizado", "mean"),
            meta=("meta_alfabetizacao_municipio_ano", "first"),
            sigla_uf=("sigla_uf", "first"),
        )
        .reset_index()
    )

    # Risco: diferença entre o que o modelo prevê e a meta oficial do
    # município — negativo significa "previsto abaixo da própria meta".
    por_municipio["gap_previsto_vs_meta"] = (
        por_municipio["prob_media_prevista"] * 100 - por_municipio["meta"]
    )

    # Só faz sentido rankear por gap onde a meta existe (municípios sem
    # meta em 2024, ou o ano-base 2023, não têm o que comparar — ver EDA).
    return por_municipio.dropna(subset=["gap_previsto_vs_meta"]).sort_values("gap_previsto_vs_meta")


def main() -> None:
    ranking = build_ranking()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT_PATH, index=False)

    cols = ["ano", "id_municipio", "sigla_uf", "n_alunos_teste", "prob_media_prevista", "meta", "gap_previsto_vs_meta"]

    print("Top 15 municípios de MAIOR risco (previsto abaixo da meta):")
    print(ranking.head(15)[cols].to_string(index=False))

    print("\nTop 15 municípios com MELHOR margem sobre a meta:")
    print(ranking.tail(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()