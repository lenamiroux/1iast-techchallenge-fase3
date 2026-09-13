"""Split treino/teste agrupado por município.

Decisão da EDA (notebooks/01_eda.ipynb): taxa_alfabetizacao_municipio é uma
agregação calculada a partir dos próprios alunos do município. Um split
aleatório por aluno deixaria o mesmo município em treino E teste,
permitindo que o modelo "memorize" a relação daquele município específico
em vez de generalizar. GroupShuffleSplit garante que nenhum município
apareça nos dois conjuntos ao mesmo tempo.
"""

from sklearn.model_selection import GroupShuffleSplit


def split_by_municipio(dataset, test_size: float = 0.2, random_state: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dataset, groups=dataset["id_municipio"]))

    train = dataset.iloc[train_idx].reset_index(drop=True)
    test = dataset.iloc[test_idx].reset_index(drop=True)

    # Verificação de sanidade: nenhum município deveria aparecer nos dois
    # conjuntos ao mesmo tempo. Se isso falhar, o split está errado.
    overlap = set(train["id_municipio"]) & set(test["id_municipio"])
    assert not overlap, f"{len(overlap)} municípios vazando entre treino e teste"

    return train, test