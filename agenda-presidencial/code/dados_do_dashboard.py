"""Contrato dos dados consumidos pelo dashboard.

Único módulo que conhece o layout em disco. ``carregar(diretorio)``
materializa um ``DadosDoDashboard`` a partir dos CSVs escritos por
gerar_dados.py. Os KPIs são funções puras sobre o dataclass — sem
filesystem, sem Plotly, sem ``date.today()`` implícito.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class DadosDoDashboard:
    compromissos: pd.DataFrame
    duracao_por_mes: pd.DataFrame
    duracao_por_dia_da_semana: pd.DataFrame
    atividades_por_hora: pd.DataFrame
    atividades_por_local: pd.DataFrame
    acumulado_diario: pd.DataFrame
    compromissos_com_projecao: pd.DataFrame
    comeco_final_do_dia: pd.DataFrame
    dias_com_compromissos: pd.DataFrame
    dia_uteis_por_mes: pd.DataFrame
    dia_uteis_por_dia_da_semana: pd.DataFrame


ARQUIVOS = {
    'compromissos':                'df.csv',
    'duracao_por_mes':             'df_duracao_por_data.csv',
    'duracao_por_dia_da_semana':   'df_duracao_por_dia_semana.csv',
    'atividades_por_hora':         'df_atividades_por_hora.csv',
    'atividades_por_local':        'df_atividades_por_local.csv',
    'acumulado_diario':            'df_acumulado.csv',
    'compromissos_com_projecao':   'completed_df.csv',
    'comeco_final_do_dia':         'df_comeco_final_dia.csv',
    'dias_com_compromissos':       'dates_with_df.csv',
    'dia_uteis_por_mes':           'df_dia_uteis.csv',
    'dia_uteis_por_dia_da_semana': 'df_dia_uteis_semana.csv',
}


def _meeting_date_para_date(df):
    df['MEETING_DATE'] = pd.to_datetime(df['MEETING_DATE'], dayfirst=True).map(lambda x: x.date())
    return df


def carregar(diretorio):
    frames = {nome: pd.read_csv(os.path.join(diretorio, arquivo))
              for nome, arquivo in ARQUIVOS.items()}
    frames['compromissos'] = _meeting_date_para_date(frames['compromissos'])
    return DadosDoDashboard(**frames)


def media_horas_por_dia(dados):
    return dados.compromissos.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean().iloc[0]


def media_horas_ultimos_30_dias(dados, hoje):
    janela = dados.compromissos[dados.compromissos['MEETING_DATE'] > (hoje - timedelta(days=30))]
    return janela.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean().iloc[0]


def dias_uteis_sem_atividades(dados):
    d = dados.dias_com_compromissos
    return int((d['MEETING_DURATION'].isnull() & (d['DIA_UTIL'] == 1)).sum())


def comparativo_clt_percentual(dados):
    ultimo = dados.acumulado_diario.tail(1)
    pct = ultimo['CUM_MEETING_DURATION'] / ultimo['CUM_DIA_UTIL'] * 100
    return int(round(pct.iloc[0]))
