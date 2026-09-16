# Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge — Fase 3 (Pós Tech FIAP, IA Scientist). Modelo supervisionado
para prever se um aluno será considerado alfabetizado, a partir da camada
Gold construída na Fase 2, com enriquecimento de fontes externas.

## Contexto do problema

A alfabetização até o 2º ano do Ensino Fundamental é a meta central do
**Compromisso Nacional Criança Alfabetizada**, com o Indicador Criança
Alfabetizada (ICA) medindo o percentual de crianças que atingem 743 pontos
na escala do Saeb. Conhecer o indicador atual não é suficiente: gestores
públicos precisam **antecipar risco**, não só descrevê-lo depois do fato.

## Objetivo analítico

Treinar um modelo supervisionado que preveja `label_alfabetizado` (1 =
alfabetizado, 0 = não) a partir de variáveis territoriais, de rede de
ensino e socioeconômicas — nunca a partir da proficiência em si, que seria
data leakage (a proficiência *é* a origem do rótulo).

## Descrição da base utilizada

Base primária: `features_alunos_ml`, camada Gold da
[Fase 2](https://github.com/lenamiroux/1iast-techchallenge-fase2)
— 3.867.999 registros de alunos avaliados em 2023/2024. Enriquecida com:

- **Censo Escolar (INEP)**, agregado por (ano, município): % de escolas
  com biblioteca, internet para aprendizagem, laboratório de informática,
  água potável, esgoto rede pública, alimentação escolar, parque infantil,
  média de alunos por turma, presença de pedagogo/psicólogo.
- **Atlas do Desenvolvimento Humano (ADH/PNUD)**, por município: IDHM,
  IDHM-Educação, renda per capita, pobreza infantil, analfabetismo adulto,
  crianças fora da escola. **Limitação importante**: única edição
  disponível é 2010 (Censo Demográfico) — usado como proxy socioeconômico
  estrutural, não como retrato contemporâneo.
- ~~FUNDEB~~: descartado — a tabela de indicadores municipais na Base dos
  Dados exige assinatura paga (BD Pro), fora do escopo gratuito do projeto.

## Bug real encontrado e corrigido (Fase 2)

Ao carregar a base pra Fase 3, `label_alfabetizado` estava **0 em 100%**
das linhas — inclusive entre alunos com proficiência ≥ 743. Causa: a
camada Gold da Fase 2 computava `(alfabetizado == "sim").cast("int")`,
assumindo de memória que a fonte usava texto `"sim"/"nao"`. O valor real,
confirmado na Silver, é string `"0"/"1"` — mesma classe de erro do bug de
`rede` já documentado na Fase 2. Corrigido em
`1iast-techchallenge-fase2/src/gold/run_gold.py` (cast direto pra int) e
reexportado antes de qualquer modelagem. Ver `notebooks/01_eda.ipynb`
para a investigação completa.

## Etapas de modelagem

1. **EDA** (`01_eda.ipynb`) — correção do rótulo, duas investigações de
   data leakage (`presenca`/`proficiencia` e `taxa_alfabetizacao_municipio`),
   hipóteses territoriais/temporais/de rede.
2. **Engenharia de atributos** (`build_features.py`) — exclusão de
   data leakage, filtro de amostra insuficiente (rede Privada, n=24), flags
   de ausência estrutural, merge do enriquecimento externo.
3. **Split treino/teste por município** (`split.py`) — `GroupShuffleSplit`,
   não split por aluno, pra evitar que o mesmo município apareça nos dois
   conjuntos (mitiga o data leakage sutil de `taxa_alfabetizacao_municipio`).
4. **Pipeline de pré-processamento** (`pipeline.py`) — `ColumnTransformer`
   com imputação (mediana) + padronização pras numéricas, one-hot pras
   categóricas, integrado ao modelo num único `Pipeline` scikit-learn.
5. **Treino e comparação de modelos** (`03_modeling_baseline.ipynb`,
   `train.py`) — Regressão Logística vs. Random Forest.
6. **Interpretabilidade** (`04_interpretability.ipynb`) — Feature
   Importance nativa + SHAP.
7. **Aplicação estratégica** (`05_aplicacao_estrategica.ipynb`,
   `municipio_risk.py`) — ranking de municípios por risco.

## Escolha do algoritmo

Testamos **Regressão Logística** (baseline interpretável) e **Random
Forest** (captura interações não-lineares). Os dois convergiram pra
**acurácia ~63%**, com Random Forest tendo recall pior pra "não
alfabetizado" (0,42 vs. 0,47). Mantivemos o **Random Forest** como modelo
final por permitir SHAP TreeExplainer eficiente para a etapa de
interpretabilidade — a diferença de performance entre os dois não
justificaria a escolha sozinha.

## Métricas de avaliação

| Métrica | Regressão Logística | Random Forest |
|---|---|---|
| Acurácia | 0,62 | 0,63 |
| Recall (não alfabetizado) | 0,46 | 0,42 |
| Recall (alfabetizado) | 0,74 | 0,77 |
| F1 macro | 0,60 | 0,59 |

Usamos `classification_report` completo (não só acurácia) porque o alvo
tem desbalanceamento moderado (59%/41%) — acurácia sozinha mascararia um
modelo que "chuta" a classe majoritária.

## Interpretação dos resultados

O SHAP confirma, de forma independente, os padrões da EDA:
`taxa_alfabetizacao_municipio` domina a predição (42% da importância);
`sigla_uf_CE` (Ceará) é o 3º fator mais importante, isolado como outlier
positivo — o modelo "redescobriu" sozinho o efeito do Paic (programa
estadual de alfabetização); `sigla_uf_BA` (Bahia) aparece como outlier
negativo. As 18 variáveis de enriquecimento carregam sinal real (direção
coerente no SHAP), mas cada uma isoladamente é pequena — parcialmente
redundante com o contexto municipal já presente na base original.

## Insights encontrados

- **O teto de ~63% de acurácia é estrutural, não de dados nem de
  algoritmo**: a maior parte das features é agregada por município (mesmo
  valor pra todos os alunos daquele lugar), e só consegue explicar
  variação *entre* municípios, nunca *entre alunos do mesmo município* —
  onde mora a maior parte da variação real do rótulo.
- **Ceará é um outlier positivo** confirmado independentemente por EDA e
  modelo — validação do Paic como referência nacional.
- **Duas lentes de risco contam histórias diferentes**: ranking por gap
  contra a meta oficial é dominado pelo Rio Grande do Sul (metas
  ambiciosas de 75-80%, não risco real); ranking por risco absoluto
  (probabilidade prevista, independente da meta) tem os 10 piores
  municípios **100% concentrados na Bahia** — achado muito mais acionável.
- **Metas municipais mal calibradas podem mascarar crise real**: alguns
  municípios baianos têm meta tão baixa (25-38%) que uma situação
  crítica (~30% de probabilidade prevista) aparece com gap *positivo*
  (bate a própria meta).

## Limitações do projeto

- **Teto de predição individual (~63% de acurácia)**: sem características
  individuais do aluno (só temos onde estuda e em que rede), o modelo não
  consegue capturar a maior parte da variação real entre alunos do mesmo
  município.
- **ADH defasado em 13-14 anos**: única edição gratuita disponível é 2010;
  tratado como proxy estrutural, não substitui um censo atualizado.
- **FUNDEB indisponível** no nível gratuito da Base dos Dados.
- **Roraima (RR) ausente** da base de alunos original (Fase 2) — nenhuma
  conclusão deste projeto se aplica a esse estado.
- **Ranking de risco cobre só o conjunto de teste** (~1.055 municípios com
  meta válida em 2024) — não é uma cobertura nacional completa; expandir
  exigiria reavaliar todos os municípios com o modelo re-treinado na base
  inteira.

## Aplicação prática para políticas públicas

- **Priorização de investimento**: o ranking por risco absoluto aponta
  concentração geográfica clara (Bahia) — mais acionável que médias
  nacionais ou regionais.
- **Auditoria de metas municipais**: municípios com meta desproporcionalmente
  baixa em relação ao risco real merecem revisão de calibração, não só
  celebração por "bater a meta".
- **Replicação de casos de sucesso**: o efeito Ceará, confirmado por dois
  métodos independentes, é candidato natural a estudo de caso para
  replicação de política pública noutros estados do Nordeste.

## Possíveis evoluções futuras

- Buscar características individuais do aluno (idade, sexo, Cadastro
  Único) pra romper o teto estrutural de predição individual.
- Testar Gradient Boosting (XGBoost/LightGBM) com ajuste de
  hiperparâmetros via `GroupKFold` (respeitando o mesmo cuidado de
  leakage por município).
- Buscar uma fonte de dados socioeconômicos municipais mais recente que o
  Censo 2010 (ex.: estimativas do IBGE Cidades, quando disponíveis a nível
  municipal).
- Expandir o ranking de risco pra cobertura nacional completa, com o
  modelo re-treinado na base inteira (não só no conjunto de teste).

## Estrutura do repositório

```
├── data/raw/                    # dados brutos e enriquecimento (não versionado)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA + correção do rótulo + hipóteses
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_interpretability.ipynb
│   └── 05_aplicacao_estrategica.ipynb
├── src/
│   ├── preprocessing/            # extração, engenharia de atributos, split, pipeline
│   ├── modeling/                 # pipeline modelo+preprocessamento, treino
│   └── evaluation/               # ranking de risco
├── reports/
│   ├── models/                   # modelo treinado (.joblib, não versionado)
│   └── municipio_risk_ranking.csv
└── requirements.txt
```

## Como rodar

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # preencher GOOGLE_CLOUD_PROJECT com um projeto GCP (BigQuery Sandbox gratuito)

python -m src.preprocessing.load_raw_data          # exporta a Gold da Fase 2
python -m src.preprocessing.fetch_external_data    # busca Censo Escolar + ADH
python -m src.modeling.train                       # treina e salva o modelo
python -m src.evaluation.municipio_risk            # gera o ranking de risco
```
