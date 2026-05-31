"""Repositório para a agenda presidencial.

Único módulo que conhece o esquema do banco. A URL vem da variável de
ambiente ``AGENDA_DB_URL``, com fallback para ``data/agenda.db`` ao lado
deste módulo. O dialeto fica a critério da URL — funciona em SQLite e
Postgres sem mudanças. Para testes, passe uma engine própria (ex.:
in-memory) no construtor.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, inspect


BASE = os.path.dirname(os.path.abspath(__file__))
URL_PADRAO = 'sqlite:///' + os.path.join(BASE, 'data', 'agenda.db')


def url_do_banco():
    return os.environ.get('AGENDA_DB_URL', URL_PADRAO)


class RepositorioAgenda:
    TABELA_COMPROMISSOS = 'AGENDA_PRESIDENCIAL'

    def __init__(self, engine=None):
        self._engine = engine if engine is not None else create_engine(url_do_banco(), echo=False)

    def carregar_compromissos(self):
        # Identificadores entre aspas — Postgres baixa-caixa não-quotados;
        # pandas.to_sql cria a tabela com aspas (preservando maiúsculas).
        return pd.read_sql_query(f'SELECT * FROM "{self.TABELA_COMPROMISSOS}"', self._engine)

    def salvar_compromissos(self, df):
        df.to_sql(self.TABELA_COMPROMISSOS, con=self._engine, if_exists='append')

    def checkpoint(self):
        """``(proximo_id, proxima_data)`` para continuar a coleta, ou ``None`` se o banco está vazio.

        SQL portável entre SQLite e Postgres — formata a data em Python e
        usa identificadores quotados para preservar maiúsculas.
        """
        if not inspect(self._engine).has_table(self.TABELA_COMPROMISSOS):
            return None
        sql = (
            f'SELECT MAX("MEETING_ID") AS "MAX_ID", MAX("BEGIN_HOUR") AS "MAX_DATE" '
            f'FROM "{self.TABELA_COMPROMISSOS}"'
        )
        result = pd.read_sql_query(sql, self._engine)
        max_date = result['MAX_DATE'][0]
        if max_date is None or pd.isna(max_date):
            return None
        proximo_id = int(result['MAX_ID'][0]) + 1
        proxima_data = (pd.to_datetime(max_date) + pd.Timedelta(days=1)).date()
        return proximo_id, proxima_data
