"""Reconstitui AGENDA_PRESIDENCIAL a partir de ``data/df.csv`` se o banco estiver vazio.

Idempotente: se ``repo.checkpoint()`` indicar dados existentes, nada acontece.
Roda no início de cada ciclo do scheduler para que ``git pull`` + boot do
container reconstrua tudo a partir do dump em git, mesmo se o volume
Postgres tiver sido apagado.
"""

import os
import sys

import pandas as pd

from repositorio import RepositorioAgenda

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PADRAO = os.path.join(BASE, 'data', 'df.csv')
COLUNAS_FONTE = ['MEETING_ID', 'BEGIN_HOUR', 'END_HOUR', 'MEETING_TITLE', 'MEETING_LOCATION']


def bootstrap(repo=None, csv_path=CSV_PADRAO):
    if repo is None:
        repo = RepositorioAgenda()
    if repo.checkpoint() is not None:
        print(f'Banco já populado (checkpoint={repo.checkpoint()}); bootstrap ignorado.')
        return 0
    if not os.path.exists(csv_path):
        print(f'Sem CSV de bootstrap em {csv_path}; banco começará vazio.')
        return 0
    df = pd.read_csv(csv_path)[COLUNAS_FONTE]
    df.set_index('MEETING_ID', inplace=True)
    repo.salvar_compromissos(df)
    print(f'Bootstrap: {len(df)} linhas carregadas de {csv_path}.')
    return len(df)


if __name__ == '__main__':
    try:
        bootstrap()
        sys.exit(0)
    except Exception as e:
        print(f'Bootstrap falhou: {e}', file=sys.stderr)
        sys.exit(1)
