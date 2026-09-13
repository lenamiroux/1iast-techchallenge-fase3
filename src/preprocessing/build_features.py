"""Constrói o conjunto de features/alvo/grupos a partir da base bruta,
aplicando as decisões justificadas na EDA (notebooks/01_eda.ipynb):
exclusão de variáveis com vazamento, remoção de amostra insuficiente,
sinalização de ausências estruturais, e enriquecimento com fontes
externas (Censo Escolar, ADH).
"""

from pathlib import Path

import pandas as pd

BASE_FEATURE_COLUMNS = [
    "ano",
    "sigla_uf",
    "rede_label",
    "taxa_alfabetizacao_municipio",
    "meta_alfabetizacao_municipio_ano",
]

# Censo Escolar: infraestrutura agregada por (ano, município), temporalmente
# alinhada com nossa base (2023/2024).
CENSO_ESCOLAR_COLUMNS = [
    "pct_biblioteca",
    "pct_sala_leitura",
    "pct_internet_aprendizagem",
    "pct_laboratorio_informatica",
    "pct_agua_potavel",
    "pct_esgoto_rede_publica",
    "pct_energia_rede_publica",
    "pct_alimentacao",
    "pct_parque_infantil",
    "media_alunos_por_turma",
    "pct_com_pedagogo",
    "pct_com_psicologo",
]

# ADH: contexto socioeconômico só de 2010 (Censo Demográfico) — proxy
# estrutural, não contemporâneo. Ver limitação documentada no README.
ADH_COLUMNS = [
    "idhm",
    "idhm_e",
    "renda_pc",
    "prop_pobreza_criancas",
    "taxa_analfabetismo_15_mais",
    "taxa_criancas_fora_escola_6_14",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CENSO_ESCOLAR_COLUMNS + ADH_COLUMNS
TARGET_COLUMN = "label_alfabetizado"
GROUP_COLUMN = "id_municipio"
WEIGHT_COLUMN = "peso_aluno"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENSO_ESCOLAR_PATH = REPO_ROOT / "data" / "raw" / "censo_escolar_municipio.parquet"
DEFAULT_ADH_PATH = REPO_ROOT / "data" / "raw" / "adh_municipio_2010.parquet"


def build_dataset(
    df: pd.DataFrame,
    censo_escolar_path: Path = DEFAULT_CENSO_ESCOLAR_PATH,
    adh_path: Path = DEFAULT_ADH_PATH,
) -> pd.DataFrame:
    """Aplica os filtros de escopo, junta o enriquecimento externo, e
    devolve um DataFrame só com as colunas que seguem para a pipeline de ML.
    """
    out = df.copy()

    # Vazamento direto: só avaliamos alunos que fizeram a prova.
    out = out[out["presenca"] == "1"]

    # Amostra insuficiente pra ser confiável (n=24 no país inteiro).
    out = out[out["rede_label"] != "Privada"]

    # Flags de ausência ANTES de imputar.
    out["taxa_municipio_ausente"] = out["taxa_alfabetizacao_municipio"].isnull().astype(int)
    out["meta_ausente"] = out["meta_alfabetizacao_municipio_ano"].isnull().astype(int)

    # Enriquecimento: Censo Escolar junta por (ano, id_municipio) — a
    # infraestrutura de um município pode mudar de um ano pro outro.
    censo_escolar = pd.read_parquet(censo_escolar_path)
    out = out.merge(censo_escolar, on=["ano", "id_municipio"], how="left")
    out["censo_escolar_ausente"] = out["pct_biblioteca"].isnull().astype(int)

    # Enriquecimento: ADH junta só por id_municipio — é uma foto única de
    # 2010, o mesmo valor se repete pra 2023 e 2024 do mesmo município.
    adh = pd.read_parquet(adh_path)
    out = out.merge(adh, on="id_municipio", how="left")
    out["adh_ausente"] = out["idhm"].isnull().astype(int)

    # Peso amostral ausente em ~0,03% dos casos (grupo raro de "presente
    # mas caderno invalidado" da EDA). Peso neutro (1.0) em vez de
    # descartar linhas com rótulo válido só por falta do peso.
    out["peso_aluno"] = out["peso_aluno"].fillna(1.0)

    flags = ["taxa_municipio_ausente", "meta_ausente", "censo_escolar_ausente", "adh_ausente"]
    keep = FEATURE_COLUMNS + flags + [TARGET_COLUMN, GROUP_COLUMN, WEIGHT_COLUMN]
    return out[keep].reset_index(drop=True)