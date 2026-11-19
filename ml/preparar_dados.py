"""Gera o recorte de São Paulo a partir das bases nacionais de SRAG (SIVEP-Gripe).

Os arquivos originais (INFLUD24 e INFLUD25, ~700 MB juntos) são baixados do
OpenDataSUS (veja dados/README.md). Este script lê só as colunas usadas, filtra
quem MORA no município de São Paulo (código IBGE 355030) e grava um CSV
compactado em dados/, pequeno o bastante para ficar no repositório.

Uso:
    python preparar_dados.py <pasta_com_os_csvs_originais>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ARQUIVOS = {
    2024: "INFLUD24-23-03-2026.csv",
    2025: "INFLUD25-16-11-2026.csv",
}

CODIGO_SAO_PAULO = "355030"  # IBGE, 6 dígitos (como vem no SIVEP-Gripe)

COLUNAS = [
    # identificação temporal e geográfica (sem dados pessoais)
    "DT_NOTIFIC", "DT_SIN_PRI", "SEM_PRI", "CO_MUN_RES",
    # perfil
    "CS_SEXO", "NU_IDADE_N", "TP_IDADE", "CS_GESTANT", "CS_RACA", "CS_ESCOL_N",
    "CS_ZONA", "NOSOCOMIAL",
    # sinais e sintomas (o que a Zela pergunta na triagem)
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA", "OUTRO_SIN",
    # fatores de risco / comorbidades
    "FATOR_RISC", "PUERPERA", "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA",
    "ASMA", "DIABETES", "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL",
    "OBESIDADE", "TABAG",
    # vacinação
    "VACINA", "VACINA_COV",
    # desfecho (alvo) e campos posteriores à entrada — só para análise, NÃO entram no modelo
    "HOSPITAL", "UTI", "SUPORT_VEN", "EVOLUCAO", "CLASSI_FIN",
]


def ler_ano(pasta: Path, ano: int) -> pd.DataFrame:
    caminho = pasta / ARQUIVOS[ano]
    print(f"Lendo {caminho.name} ...", flush=True)
    pedacos = []
    for pedaco in pd.read_csv(
        caminho,
        sep=";",
        usecols=COLUNAS,
        dtype=str,
        encoding="utf-8",
        chunksize=200_000,
    ):
        pedacos.append(pedaco[pedaco["CO_MUN_RES"] == CODIGO_SAO_PAULO])
    df = pd.concat(pedacos, ignore_index=True)
    df.insert(0, "ANO_BASE", ano)
    print(f"  {len(df):,} notificações de residentes em São Paulo", flush=True)
    return df


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pasta = Path(sys.argv[1])
    df = pd.concat([ler_ano(pasta, ano) for ano in ARQUIVOS], ignore_index=True)

    saida = Path(__file__).parent / "dados" / "srag_sao_paulo_2024_2025.csv.gz"
    saida.parent.mkdir(exist_ok=True)
    df.to_csv(saida, index=False, compression="gzip")
    print(f"Gravado {saida} ({saida.stat().st_size / 1e6:.1f} MB, {len(df):,} linhas)")


if __name__ == "__main__":
    main()
