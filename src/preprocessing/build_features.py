"""Constrói o conjunto de features/alvo/grupos a partir da base bruta,
aplicando as decisões justificadas na EDA (notebooks/01_eda.ipynb):
exclusão de variáveis com vazamento, remoção de amostra insuficiente,
e sinalização de ausências estruturais.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "ano",
    "sigla_uf",
    "rede_label",
    "taxa_alfabetizacao_municipio",
    "meta_alfabetizacao_municipio_ano",
]
TARGET_COLUMN = "label_alfabetizado"
GROUP_COLUMN = "id_municipio"  # usado no split (Etapa 6), não é feature
WEIGHT_COLUMN = "peso_aluno"


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica os filtros de escopo e devolve um DataFrame só com as colunas
    que seguem para a pipeline de ML (features + alvo + grupo + peso).
    """
    out = df.copy()

    # Vazamento direto: só avaliamos alunos que fizeram a prova. Ausentes
    # têm label determinado pela própria ausência, não por um padrão a
    # aprender (ver EDA, Hipótese sobre `presenca`).
    out = out[out["presenca"] == "1"]

    # Amostra insuficiente pra ser confiável (n=24 no país inteiro).
    out = out[out["rede_label"] != "Privada"]

    # Flags de ausência ANTES de imputar — a ausência em si é sinal
    # territorial (município sem meta/dado agregado), não deve ser escondida.
    out["taxa_municipio_ausente"] = out["taxa_alfabetizacao_municipio"].isnull().astype(int)
    out["meta_ausente"] = out["meta_alfabetizacao_municipio_ano"].isnull().astype(int)

    # Peso amostral ausente em ~0,03% dos casos (mesmo grupo raro de
    # "presente mas caderno invalidado" da EDA). Peso neutro (1.0) em vez
    # de descartar linhas com rótulo válido só por falta do peso.
    out["peso_aluno"] = out["peso_aluno"].fillna(1.0)

    keep = FEATURE_COLUMNS + ["taxa_municipio_ausente", "meta_ausente", TARGET_COLUMN, GROUP_COLUMN, WEIGHT_COLUMN]
    return out[keep].reset_index(drop=True)