# Dados

## Fonte

| | |
| --- | --- |
| **Base** | Banco de dados da Síndrome Respiratória Aguda Grave (SRAG), 2019 a 2026, do SIVEP-Gripe |
| **Responsável** | Ministério da Saúde (CGIAE/DAENT/SVSA), publicado pelo DATASUS |
| **Página oficial** | <https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026> |
| **Arquivos usados** | `INFLUD24-23-03-2026.csv`: ano epidemiológico **2024**, base congelada (versão de 23/03/2026) · `INFLUD25-16-11-2026.csv`: ano epidemiológico **2025**, base viva (versão de 16/11/2026) |
| **Links diretos** | <https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-23-03-2026.csv> · <https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-16-11-2026.csv> |
| **Dicionário** | <https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/dicionario-de-dados-2019-a-2025.pdf> |
| **Período** | Início dos sintomas de 31/12/2023 a 03/01/2026 (semanas epidemiológicas de 2024 e 2025) |
| **Acesso** | 18/11/2026 |
| **Privacidade** | Dados **reais e anonimizados** pelo Ministério da Saúde (LGPD, Lei 13.709/2018). Não há nome, CPF nem endereço. |

O SIVEP-Gripe é o sistema oficial em que hospitais e unidades notificam os casos de SRAG (internação por síndrome
respiratória grave) e os óbitos, inclusive por influenza, covid-19 e vírus sincicial respiratório.

## Recorte usado: `srag_sao_paulo_2024_2025.csv.gz`

Gerado por [`../preparar_dados.py`](../preparar_dados.py) a partir dos dois CSVs nacionais (cerca de 700 MB, que
não ficam no repositório):

```bash
python preparar_dados.py <pasta_com_os_csvs_baixados>
```

- **Filtro:** pessoas que **moram** no município de São Paulo (`CO_MUN_RES = 355030`, código IBGE).
- **Resultado:** 35.080 notificações (17.408 de 2024 e 17.672 de 2025), 47 colunas, 0,5 MB compactado.
- **Colunas mantidas:** datas de notificação e de início dos sintomas, perfil (sexo, idade, gestação, raça,
  escolaridade, zona), sinais e sintomas, fatores de risco e comorbidades, vacinação e os desfechos (internação, UTI,
  suporte ventilatório, evolução e classificação final). Raça, escolaridade, UTI, suporte ventilatório e classificação
  final ficam no arquivo para conferência, mas **não entram no modelo** (ver o notebook).

## Dicionário das colunas usadas no modelo

Nas perguntas de sim/não do SIVEP-Gripe: `1` = sim, `2` = não, `9` = ignorado, vazio = não preenchido.

| Coluna | Significado | Como foi usada |
| --- | --- | --- |
| `EVOLUCAO` | 1 cura · 2 óbito por SRAG · 3 óbito por outras causas · 9 ignorado | **Alvo**: 1 = óbito por SRAG, 0 = cura (3, 9 e vazio saem) |
| `ANO_BASE` | Ano epidemiológico do arquivo de origem (criada pelo script) | Divisão treino (2024) / teste (2025) |
| `NU_IDADE_N` + `TP_IDADE` | Idade e unidade (1 dias · 2 meses · 3 anos) | `idade_anos` |
| `CS_SEXO` | M · F · I (ignorado) | `sexo_masculino` |
| `CS_GESTANT` | 1–4 gestante (trimestre ou idade gestacional ignorada) · 5 não · 6 não se aplica · 9 ignorado | `gestante` |
| `PUERPERA` | Puérpera | comorbidade |
| `FEBRE`, `TOSSE`, `GARGANTA`, `DISPNEIA`, `DESC_RESP`, `SATURACAO`, `DIARREIA`, `VOMITO`, `DOR_ABD`, `FADIGA`, `PERD_OLFT`, `PERD_PALA` | Sinais e sintomas (`SATURACAO` = saturação de O₂ < 95%) | 1/0, ignorado = faltante |
| `FATOR_RISC` | Possui fator de risco | Comorbidades só valem quando marcado |
| `CARDIOPATI`, `HEMATOLOGI`, `SIND_DOWN`, `HEPATICA`, `ASMA`, `DIABETES`, `NEUROLOGIC`, `PNEUMOPATI`, `IMUNODEPRE`, `RENAL`, `OBESIDADE` | Doenças pré-existentes | 1/0 (0 quando não há fator de risco) |
| `VACINA` | Vacinado contra gripe na última campanha | 1/0 |
| `VACINA_COV` | Recebeu vacina contra covid-19 | 1/0 |
| `TABAG` | Tabagismo | **Fora**: 100% vazio em 2024 e 2025 |
