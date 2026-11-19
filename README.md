# Zela · Machine Learning: sinais de risco de óbito em casos respiratórios graves

Parte de **Inteligência Artificial & Machine Learning** da **Zela**, a CitizenTech de triagem e orientação em saúde
da nossa equipe. Usamos **dados reais e públicos do SUS** (SIVEP-Gripe, 2024 e 2025, moradores de São Paulo) para
descobrir quais sinais que a Zela pergunta na triagem indicam risco de morte num caso respiratório grave.

> **FIAP · Global Solution 2026.2 · CitizenTech · Engenharia de Software (3º ano)**
> Disciplina: **Inteligência Artificial & Machine Learning**

## Integrantes

| Nome | RM |
| --- | --- |
| João Vitor Carotta Ribeiro | RM 555187 |
| Arthur Bueno de Oliveira | RM 558396 |
| Victor Magdaleno Marcos | RM 556729 |

## Ficha da CitizenTech

| | |
| --- | --- |
| **Nome** | Zela |
| **Cidade e bairro** | São Paulo, Vila Andrade (Paraisópolis) |
| **Cidadão beneficiado** | Morador de comunidade com pouco acesso à saúde, baixa alfabetização e internet ruim. Persona: Regina, 53 anos, na fila do SUS sem previsão de atendimento. |
| **Problema, em uma frase** | A pessoa não sabe se o caso é leve, se deve ir à UBS ou se é emergência. |
| **Como funciona** | Triagem por voz, foto ou conversa que classifica em verde (casa), amarelo (UBS) ou vermelho (SAMU 192). Só vai ao humano quem precisa. |
| **O que cada parte faz** | API da Zela (Python/FastAPI): triagem e encaminhamento · App (Expo) e site: porta de entrada do cidadão · Zela Gestão (C#): fila da UBS e visitas do agente comunitário · **ML (este repositório)**: mostra, com dados reais, quais sinais indicam risco de óbito, para calibrar o vermelho da triagem. |
| **Repositório principal** | <https://github.com/JoaoVitorCarottaRibeiro/zela> |
| **Indicador de impacto** | Óbitos que chegariam ao vermelho (recall do alerta: 77% em 2025) e casos graves encaminhados sem espera. |
| **ODS** | **3 Saúde e bem-estar** (identificar cedo quem corre risco) · **10 Redução das desigualdades** (triagem por voz para quem não lê) · 9 Inovação (dados abertos do SUS a serviço da comunidade). |

## Pergunta de ML

> Entre moradores de São Paulo notificados com Síndrome Respiratória Aguda Grave, é possível prever, só com o que se
> sabe na chegada (sintomas, idade, gestação, doenças pré-existentes e vacinação), quem evolui para óbito? E quais
> sinais pesam mais?

## Resultado em 30 segundos

- **Modelo:** XGBoost, escolhido entre 36 combinações de `max_depth`, `learning_rate` e `n_estimators` com validação
  cruzada de 5 partes só no treino.
- **Treino com 2024, teste com 2025:** ROC-AUC **0,93 / 0,90 / 0,87** (treino / validação / teste), com overfitting
  pequeno e controlado.
- **Alerta:** sinaliza 26% das pessoas e encontra **77% dos óbitos** de 2025. Na faixa de maior risco, 28% morreram;
  na de menor, 0,4%.
- **Para a Zela:** idade é o fator mais forte, seguida de desconforto respiratório, falta de ar e saturação baixa.
  Recomendação: **60 anos ou mais com falta de ar sobe para vermelho**.

## Entregáveis

| Arquivo | O que é |
| --- | --- |
| [`ml/zela_srag_obito.ipynb`](ml/zela_srag_obito.ipynb) | Notebook **executado**, do problema à conclusão |
| [`RELATORIO_ML.md`](RELATORIO_ML.md) | Relatório: problema, dados, modelo, treino, resultados e conclusão |
| [`ml/dados/README.md`](ml/dados/README.md) | Fonte, link, período e dicionário dos dados |
| [`ml/dados/srag_sao_paulo_2024_2025.csv.gz`](ml/dados) | Recorte de São Paulo usado no notebook (0,5 MB) |
| [`ml/requirements.txt`](ml/requirements.txt) | Bibliotecas e versões |
| [`ml/preparar_dados.py`](ml/preparar_dados.py) | Gera o recorte a partir das bases nacionais do OpenDataSUS |
| [`ml/gerar_notebook.py`](ml/gerar_notebook.py) | Fonte legível das células do notebook |
| [`ml/figuras/`](ml/figuras) | Gráficos gerados pelo notebook |
| **Vídeo** | _link do YouTube (não listado), até 5 min_ |

## Como rodar

Pré-requisito: Python 3.12 ou mais novo.

```bash
git clone https://github.com/victormmarcos/zela-ml.git
cd zela-ml/ml
python -m venv .venv
.venv\Scripts\activate              # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook zela_srag_obito.ipynb
```

Ou, para executar tudo de uma vez:
`jupyter nbconvert --to notebook --execute --inplace zela_srag_obito.ipynb` (cerca de 2 minutos).

## Dados

**Reais e públicos:** SIVEP-Gripe, Banco de SRAG 2019 a 2026, Ministério da Saúde / OpenDataSUS
(<https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026>), bases de 2024 (versão 23/03/2026) e 2025 (versão
16/11/2026), filtradas para residentes de São Paulo. Os dados são anonimizados pelo Ministério da Saúde (LGPD). Nenhum
dado simulado foi usado. Detalhes em [`ml/dados/README.md`](ml/dados/README.md).
