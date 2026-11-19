# Relatório de Machine Learning · Zela

**Quais sinais da triagem indicam risco de morte num caso respiratório grave?**

FIAP · Global Solution 2026.2 · CitizenTech · Inteligência Artificial & Machine Learning
Equipe: João Vitor Carotta Ribeiro (RM 555187) · Arthur Bueno de Oliveira (RM 558396) · Victor Magdaleno Marcos (RM 556729)
Notebook executado: [`ml/zela_srag_obito.ipynb`](ml/zela_srag_obito.ipynb)

---

## 1. Problema e pergunta

A **Zela** é a nossa CitizenTech de triagem em saúde para a Vila Andrade (Paraisópolis), em São Paulo. O morador conta
o que sente por voz, foto ou conversa e recebe um nível: **verde** (cuidar em casa), **amarelo** (UBS com prioridade)
ou **vermelho** (emergência, SAMU 192). A regra de ouro do projeto é que um sinal de alerta **nunca** é rebaixado.
Hoje os sinais vermelhos são uma lista fixa de palavras ("falta de ar", "lábios roxos", "dor no peito"…).

**Pergunta de ML:** entre moradores de São Paulo notificados com Síndrome Respiratória Aguda Grave (SRAG), é possível
prever, **só com o que se sabe na chegada** (sintomas, idade, gestação, doenças pré-existentes e vacinação), quem
**evolui para óbito**? E quais sinais pesam mais?

É uma classificação binária (óbito por SRAG = 1, cura = 0). A resposta diz à Zela, com dados reais, **quais relatos
precisam ir direto para o vermelho**.

> Começamos tentando prever a **ida para a UTI**, mas o modelo ficou fraco (ROC-AUC 0,58 no teste). Entre pessoas já
> internadas, os sintomas quase não separam quem vai para a UTI, e essa decisão depende de vaga e de cada hospital.
> O óbito é um desfecho mais objetivo e é exatamente o que o vermelho da Zela quer evitar.

## 2. Dados reais e públicos

| | |
| --- | --- |
| **Fonte** | SIVEP-Gripe, Banco de SRAG 2019 a 2026, Ministério da Saúde / OpenDataSUS |
| **Link** | <https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026> |
| **Arquivos** | `INFLUD24` (2024, base congelada de 23/03/2026) e `INFLUD25` (2025, base viva de 16/11/2026) |
| **Período** | Início dos sintomas de 31/12/2023 a 03/01/2026 |
| **Recorte** | Residentes do município de São Paulo (IBGE 355030): **35.080 notificações** |
| **Com desfecho conhecido** | **33.875** casos (cura ou óbito por SRAG), com **2.650 óbitos** (7,8%) |

Os dados são anonimizados pelo Ministério da Saúde (LGPD). Fonte, link, dicionário e o script que gera o recorte
estão em [`ml/dados/README.md`](ml/dados/README.md).

## 3. Preparação e divisão treino/teste

- **Codificação:** as perguntas do SIVEP usam 1 = sim, 2 = não, 9 = ignorado. Viraram 1/0, e "ignorado" ou vazio
  virou **faltante**, sem inventar valor (o XGBoost aprende o melhor caminho para o faltante). A idade foi convertida
  para anos (o SIVEP mistura dias, meses e anos). Duas variáveis-resumo: quantidade de sintomas e de comorbidades.
  Total: **31 variáveis**.
- **Comorbidades:** só são preenchidas quando "fator de risco" é marcado. Sem fator de risco, consideramos "não tem".
- **Fora do modelo, por vazamento:** UTI, suporte ventilatório, classificação final do vírus e datas de internação.
  Só existem depois da chegada ao hospital e fariam o modelo "colar".
- **Fora do modelo, por ética:** raça/cor e escolaridade. A Zela não pergunta isso, e o modelo não deve priorizar nem
  rebaixar ninguém por esses atributos.
- **Fora do modelo, por defeito:** tabagismo (`TABAG`) vem 100% vazio nas duas bases. Do jeito que as comorbidades
  são codificadas, viraria um disfarce de "tem fator de risco".
- **Divisão temporal:** **treino = 2024** (16.662 casos, 7,2% de óbito) e **teste = 2025** (17.213 casos, 8,4%).
  É o uso real (aprender com o passado, prever o futuro) e testa se o modelo aguenta a mudança de vírus e de estação
  de um ano para o outro. Validação cruzada, escolha de hiperparâmetros e ponto de corte usam **só 2024**.

## 4. Escolha do modelo

Escolhemos o **XGBoost**:

- os dados são uma tabela de variáveis sim/não, contagens e idade, terreno em que árvores com *boosting* costumam ser
  as mais fortes;
- trata **faltantes** de forma nativa (até 39% de "ignorado" em alguns sintomas);
- captura **interações**, como falta de ar pesando mais em idosos;
- permite medir a importância de cada sinal.

Na validação cruzada (5 partes, só 2024), comparamos com duas referências:

| Modelo | ROC-AUC treino | ROC-AUC validação | PR-AUC validação |
| --- | --- | --- | --- |
| Sempre "não evolui para óbito" | 0,500 | 0,500 | 0,072 |
| Regressão logística | 0,867 | 0,863 | 0,334 |
| XGBoost (padrão) | 0,987 | 0,874 | 0,449 |

O XGBoost ganha, mas o padrão já mostra **overfitting** (0,99 no treino), o que motivou a busca de hiperparâmetros.

**Métricas:** os óbitos são poucos (7–8%), e um modelo que diz "ninguém morre" acerta 92%. Por isso não usamos
acurácia. A principal é a **ROC-AUC** (o quanto o modelo ordena bem quem tem mais risco). Também olhamos a
**PR-AUC**, o **recall** (dos que morreram, quantos seriam sinalizados) e a **precisão**.

## 5. Treino com diferentes hiperparâmetros

Testamos **36 combinações** com `GridSearchCV` e validação cruzada estratificada de 5 partes **só no treino**:
`max_depth` ∈ {2, 4, 6, 8} × `learning_rate` ∈ {0,03; 0,1; 0,3} × `n_estimators` ∈ {100, 300, 600}.

| max_depth | learning_rate | n_estimators | AUC treino | AUC validação | diferença |
| --- | --- | --- | --- | --- | --- |
| **4** | **0,03** | **300** | **0,929** | **0,897** | **0,032** ← escolhida |
| 2 | 0,03 | 600 | 0,909 | 0,895 | 0,014 |
| 6 | 0,03 | 300 | 0,957 | 0,892 | 0,065 |
| 8 | 0,1 | 600 | 0,999 | 0,859 | 0,140 |
| 8 | 0,3 | 600 | 1,000 | 0,843 | 0,157 ← pior |

Árvores mais profundas e passos maiores **decoram o treino** (AUC perto de 1,0) e **pioram na validação**. A
figura `ml/figuras/03_overfitting_max_depth.png` mostra a curva de complexidade. A combinação escolhida tem a melhor
validação com diferença pequena entre treino e validação.

**Ponto de corte:** como perder um caso grave é pior que um alarme a mais, escolhemos (nas previsões fora da dobra
do treino) o maior corte que ainda sinaliza **pelo menos 80% dos óbitos**: probabilidade ≥ **0,10**.

## 6. Resultados em treino e teste

| Métrica | Treino (2024) | Validação cruzada (2024) | **Teste (2025)** |
| --- | --- | --- | --- |
| ROC-AUC | 0,925 | 0,896 | **0,866** |
| PR-AUC | 0,599 | 0,489 | **0,411** |
| Recall (óbitos sinalizados) | 0,872 | 0,811 | **0,770** |
| Precisão (alertas certos) | 0,282 | 0,261 | **0,251** |
| F1 | 0,426 | 0,395 | **0,379** |
| Taxa real de óbito | 7,2% | 7,2% | 8,4% |

- **Overfitting controlado:** a queda de treino → validação → teste é pequena (0,93 → 0,90 → 0,87). O pouco que cai
  no teste é esperado num ano diferente, com outra mistura de vírus e taxa de óbito maior (8,4% contra 7,2%).
- **Na prática (2025):** o alerta marca 4.438 de 17.213 pessoas (26%) e encontra **77% dos óbitos**. Entre os
  marcados, 25% morreram, **3 vezes a média**.
- **Faixas de risco** (quintis da probabilidade no teste): a taxa real de óbito sobe de **0,4%** na faixa de menor
  risco para **28,1%** na de maior risco (figura `06_faixas_de_risco.png`).

## 7. Conclusão: o que isso significa para o cidadão

1. **A idade é o sinal mais forte** (a maior queda de AUC na importância por permutação). Em 2025, 20,4% das pessoas
   com 60 anos ou mais morreram, contra 1,2% de quem tinha menos de 60 anos sem falta de ar e sem saturação baixa.
2. **Depois vêm os sinais respiratórios:** desconforto respiratório, falta de ar e saturação baixa. Falta de ar com
   desconforto: 12,3% de óbito, contra 8,4% no geral. Isso **confirma** os sinais vermelhos que a Zela já usa.
3. **Doenças crônicas somam risco:** diabetes, doença pulmonar e cardíaca aumentam o óbito em cerca de 10 pontos
   percentuais cada (análise descritiva).
4. **Febre e tosse sozinhas não indicam gravidade:** aparecem com taxa de óbito menor. Elas são comuns em crianças
   pequenas com vírus respiratórios, um grupo que morre pouco. Por isso a Zela não deve subir o nível só por febre.

**Recomendação para a Zela:** manter os sinais respiratórios como vermelho e acrescentar uma regra de idade. Uma
pessoa **com 60 anos ou mais que relata falta de ar ou cansaço para respirar sobe de nível**, mesmo sem outra
palavra de alarme. No app, a frase para o morador fica simples: *"Por causa da sua idade e da falta de ar, procure
atendimento agora. Se piorar, ligue 192."*

## 8. Limitações

- A base só tem casos **notificados como SRAG**, ou seja, pessoas que chegaram a um serviço de saúde. O modelo não
  diz se alguém precisa procurar ajuda; ele mostra **quais sinais indicam gravidade** entre os casos respiratórios
  graves. Por isso, na Zela, ele só pode **subir** a prioridade, nunca rebaixar (a regra de ouro continua valendo).
- Muitos campos vêm como "ignorado" (até 39%), e o preenchimento varia entre hospitais e anos.
- A base de 2025 ainda é "viva": casos em aberto ficaram de fora e alguns desfechos podem ser atualizados.
- A precisão de 25% significa que 3 em cada 4 alertas não terminam em óbito. Para triagem isso é aceitável (o alerta
  leva a atendimento rápido, não a um tratamento invasivo), mas não serve como diagnóstico.

## 9. Como reproduzir

```bash
cd ml
python -m venv .venv && .venv\Scripts\activate      # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace zela_srag_obito.ipynb
```

O recorte de São Paulo já está no repositório. Para gerá-lo de novo a partir das bases nacionais, veja
`ml/dados/README.md`. Todas as etapas usam semente fixa (42), então os números se repetem.
