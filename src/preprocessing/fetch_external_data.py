"""Busca e agrega fontes externas (Censo Escolar, ADH) pra enriquecer a
base de modelagem, seguindo a sugestão do próprio desafio. Salva
localmente em Parquet — não versiona o dado, só o processo de obtê-lo
(mesmo padrão de load_raw_data.py).
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.bigquery import QueryJobConfig

load_dotenv()

OUTPUT_DIR = Path("data/raw")

CENSO_ESCOLAR_QUERY = """
SELECT
  ano,
  id_municipio,
  COUNT(*) AS n_escolas,
  AVG(biblioteca) AS pct_biblioteca,
  AVG(sala_leitura) AS pct_sala_leitura,
  AVG(internet_aprendizagem) AS pct_internet_aprendizagem,
  AVG(laboratorio_informatica) AS pct_laboratorio_informatica,
  AVG(agua_potavel) AS pct_agua_potavel,
  AVG(esgoto_rede_publica) AS pct_esgoto_rede_publica,
  AVG(energia_rede_publica) AS pct_energia_rede_publica,
  AVG(alimentacao) AS pct_alimentacao,
  AVG(parque_infantil) AS pct_parque_infantil,
  SAFE_DIVIDE(
    SUM(quantidade_matricula_fundamental_anos_iniciais),
    NULLIF(SUM(quantidade_turma_fundamental_anos_iniciais), 0)
  ) AS media_alunos_por_turma,
  AVG(CASE WHEN profissional_pedagogia > 0 THEN 1 ELSE 0 END) AS pct_com_pedagogo,
  AVG(CASE WHEN profissional_psicologo > 0 THEN 1 ELSE 0 END) AS pct_com_psicologo
FROM `basedosdados.br_inep_censo_escolar.escola`
WHERE ano IN (2023, 2024)
  AND etapa_ensino_fundamental_anos_iniciais = 1
GROUP BY ano, id_municipio
"""

# ADH só tem edição de 2010 (Censo Demográfico) — usado como proxy
# estrutural, não como dado contemporâneo. Documentar essa defasagem no
# README é obrigatório, não opcional.
ADH_QUERY = """
SELECT
  id_municipio,
  idhm,
  idhm_e,
  renda_pc,
  prop_pobreza_criancas,
  taxa_analfabetismo_15_mais,
  taxa_criancas_fora_escola_6_14
FROM `basedosdados.mundo_onu_adh.municipio`
WHERE ano = 2010
"""


def _run_query(client: bigquery.Client, query: str, label: str) -> pd.DataFrame:
    dry_run = client.query(query, job_config=QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = dry_run.total_bytes_processed / (1024**3)
    print(f"FinOps: {label} vai escanear ~{gb:.3f} GB (BigQuery Sandbox: 1 TB/mês grátis)")
    return client.query(query).to_dataframe()


def main() -> None:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT não definido no .env")
    client = bigquery.Client(project=project)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    censo = _run_query(client, CENSO_ESCOLAR_QUERY, "Censo Escolar (agregado)")
    censo.to_parquet(OUTPUT_DIR / "censo_escolar_municipio.parquet", index=False)
    print(f"Censo Escolar: {len(censo)} linhas (ano x município) -> {OUTPUT_DIR / 'censo_escolar_municipio.parquet'}")

    adh = _run_query(client, ADH_QUERY, "ADH (município, 2010)")
    adh.to_parquet(OUTPUT_DIR / "adh_municipio_2010.parquet", index=False)
    print(f"ADH: {len(adh)} linhas (município) -> {OUTPUT_DIR / 'adh_municipio_2010.parquet'}")


if __name__ == "__main__":
    main()