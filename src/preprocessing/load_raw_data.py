"""Consolida os parquets particionados da camada Gold (Fase 2) num único
arquivo local, servindo de ponto de entrada reproduzível pra esta pipeline
de ML. Não versiona o dado em si — só o processo de obtê-lo.
"""

import argparse
import glob
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path.home() / "code" / "1iast-techchallenge-fase2" / "data" / "gold" / "features_alunos_ml"
DEFAULT_OUTPUT = Path("data/raw/features_alunos_ml.parquet")


def load_and_consolidate(source_dir: Path, output_path: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(source_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"Nenhum .parquet encontrado em {source_dir}. "
            "Rode a pipeline da Fase 2 (make pipeline) antes de extrair os dados."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida a camada Gold da Fase 2 para uso em ML")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_and_consolidate(args.source, args.output)
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(f"Salvo em: {args.output}")


if __name__ == "__main__":
    main()