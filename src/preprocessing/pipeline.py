"""ColumnTransformer de pré-processamento — separado do modelo (Etapa 7),
pra poder ser reusado com qualquer classificador e testado isoladamente.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Colunas numéricas com ausência estrutural (ver EDA) — imputamos com a
# mediana (robusta a outliers) e padronizamos (StandardScaler), técnica de
# transformação exigida pelo desafio mesmo que o modelo final seja uma
# árvore (que não precisa de escala) — mantém a pipeline agnóstica ao
# algoritmo escolhido depois.
NUMERIC_WITH_MISSING = ["taxa_alfabetizacao_municipio", "meta_alfabetizacao_municipio_ano"]

# Já vêm completas (sem nulo) — não precisam de imputação, só padronização
# pra manter a mesma escala das outras numéricas.
NUMERIC_COMPLETE = ["ano", "taxa_municipio_ausente", "meta_ausente"]

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