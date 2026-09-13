"""ColumnTransformer de pré-processamento — separado do modelo (Etapa 7),
pra poder ser reusado com qualquer classificador e testado isoladamente.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numéricas com ausência estrutural — as duas originais da camada Gold
# (taxa/meta municipal, ver EDA) mais o enriquecimento externo (Censo
# Escolar e ADH). Mesmo o Censo Escolar tendo 0% de ausência nos dados
# atuais e o ADH só 0,04%, mantemos todas aqui por segurança — uma
# atualização futura da fonte pode ter cobertura diferente, e o imputer
# não quebra mesmo que a ausência real seja zero. Imputamos com a mediana
# (robusta a outliers) e padronizamos (StandardScaler) — transformação
# exigida pelo desafio mesmo que o modelo final seja uma árvore (que não
# precisa de escala), mantendo a pipeline agnóstica ao algoritmo escolhido
# depois.
NUMERIC_WITH_MISSING = [
    "taxa_alfabetizacao_municipio",
    "meta_alfabetizacao_municipio_ano",
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
    "idhm",
    "idhm_e",
    "renda_pc",
    "prop_pobreza_criancas",
    "taxa_analfabetismo_15_mais",
    "taxa_criancas_fora_escola_6_14",
]

# Já vêm completas (sem nulo) — o ano da avaliação e as flags binárias de
# ausência (uma por fonte: base original, Censo Escolar, ADH). Não
# precisam de imputação, só padronização pra manter a mesma escala das
# outras numéricas.
NUMERIC_COMPLETE = ["ano", "taxa_municipio_ausente", "meta_ausente", "censo_escolar_ausente", "adh_ausente"]

# Categóricas de baixa cardinalidade — one-hot é seguro aqui (27 UFs, 2
# redes). handle_unknown="ignore" evita quebrar em produção se aparecer
# uma categoria nova (ex.: uma UF nunca vista no treino).
CATEGORICAL = ["sigla_uf", "rede_label"]


def build_preprocessor() -> ColumnTransformer:
    numeric_missing_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    numeric_complete_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num_missing", numeric_missing_pipeline, NUMERIC_WITH_MISSING),
        ("num_complete", numeric_complete_pipeline, NUMERIC_COMPLETE),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])