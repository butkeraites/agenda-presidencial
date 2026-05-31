"""Agregações puras sobre compromissos e calendário.

Cada função recebe DataFrames e devolve DataFrames: sem banco, sem
filesystem, sem rede. O pipeline (gerar_dados.py) liga as peças.

Os nomes das colunas permanecem em UPPER_SNAKE_CASE para preservar o
contrato em disco já consumido por dashboard.py.
"""

from datetime import timedelta

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


ORDEM_DIA_SEMANA = [
    "Domingo", "Segunda-feira", "Terça-feira", "Quarta-feira",
    "Quinta-feira", "Sexta-feira", "Sábado",
]


def hora_arredondada(t):
    t_rounded = t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30)
    return t_rounded.hour


def compromissos_com_duracao(df):
    df = df.copy()
    df['BEGIN_HOUR'] = pd.to_datetime(df['BEGIN_HOUR'])
    df['ROUNDED_BEGIN_HOUR'] = df['BEGIN_HOUR'].map(hora_arredondada)
    df['END_HOUR'] = pd.to_datetime(df['END_HOUR'])
    df['ROUNDED_END_HOUR'] = df['END_HOUR'].map(hora_arredondada)
    df['MEETING_DATE'] = df['BEGIN_HOUR'].map(lambda x: x.date())
    df['MEETING_DURATION'] = (df['END_HOUR'] - df['BEGIN_HOUR']).map(lambda x: x.total_seconds() / 3600)
    return df


def compromissos_com_calendario(compromissos, calendario):
    return compromissos.merge(calendario, on='MEETING_DATE', how='left')


def duracao_por_mes(compromissos_com_cal):
    return (
        compromissos_com_cal
        .groupby('MES_REFERENCIA')
        .sum()[['MEETING_DURATION']]
        .reset_index()
    )


def dia_uteis_por_mes(calendario, ate_mes):
    df = calendario.groupby('MES_REFERENCIA').sum()[['DIA_UTIL']]
    df['HORAS_DE_TRABALHO'] = df['DIA_UTIL'] * 8
    df = df.reset_index()
    return df.loc[df['MES_REFERENCIA'] <= ate_mes]


def duracao_por_dia_da_semana(compromissos_com_cal):
    df = compromissos_com_cal.groupby('DIA_DA_SEMANA').sum()[['MEETING_DURATION']].reset_index()
    df['DIA_DA_SEMANA'] = pd.CategoricalIndex(df['DIA_DA_SEMANA'], ordered=True, categories=ORDEM_DIA_SEMANA)
    return df.sort_values('DIA_DA_SEMANA')


def dia_uteis_por_dia_da_semana(calendario, ate_data):
    df = (
        calendario.loc[calendario['MEETING_DATE'] <= ate_data]
        .groupby('DIA_DA_SEMANA')
        .sum()[['DIA_UTIL']]
    )
    df['HORAS_DE_TRABALHO'] = df['DIA_UTIL'] * 8
    df = df.reset_index()
    df['DIA_DA_SEMANA'] = pd.CategoricalIndex(df['DIA_DA_SEMANA'], ordered=True, categories=ORDEM_DIA_SEMANA)
    return df.sort_values('DIA_DA_SEMANA')


def atividades_por_hora(compromissos):
    return compromissos.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_ID'].count().reset_index()


def atividades_por_local(compromissos):
    return compromissos.groupby(['MEETING_LOCATION'])['MEETING_ID'].count().reset_index()


def acumulado_diario(compromissos, calendario, hoje):
    """Retorna (df_acumulado, dates_with_df) — o dashboard consome ambos."""
    diario = compromissos.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].reset_index()
    com_dia_util = calendario.merge(diario, on='MEETING_DATE', how='left')
    dates_with_df = com_dia_util[com_dia_util['MEETING_DATE'] <= hoje]
    acumulado = dates_with_df[['MEETING_DATE', 'MEETING_DURATION', 'DIA_UTIL']].fillna(0)
    acumulado = pd.concat([acumulado, acumulado[['MEETING_DURATION', 'DIA_UTIL']].cumsum().add_prefix('CUM_')], axis=1)
    acumulado['CUM_DIA_UTIL'] = acumulado['CUM_DIA_UTIL'] * 8
    return acumulado, dates_with_df


def projecao_2d_titulos(compromissos, stop_words):
    tfidf = TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1, 4), stop_words=stop_words)
    matriz = tfidf.fit_transform(list(compromissos['MEETING_TITLE']))
    densa = pd.DataFrame(matriz.toarray(), columns=tfidf.get_feature_names())
    reduzida = pd.DataFrame(data=PCA(n_components=2).fit_transform(densa))
    reduzida = reduzida.add_prefix('DIMENSION_').reset_index()
    return compromissos.reset_index().merge(reduzida, on='index', how='left').drop(['index'], axis=1)


def primeira_ultima_atividade_do_dia(compromissos):
    primeira = compromissos.groupby('MEETING_DATE').agg({'ROUNDED_BEGIN_HOUR': 'min'}).reset_index()
    primeira = primeira.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_DATE'].count().reset_index()
    ultima = compromissos.groupby('MEETING_DATE').agg({'ROUNDED_END_HOUR': 'max'}).reset_index()
    ultima = ultima.groupby(['ROUNDED_END_HOUR'])['MEETING_DATE'].count().reset_index()
    primeira['DAY_HOUR'] = primeira['ROUNDED_BEGIN_HOUR']
    ultima['DAY_HOUR'] = ultima['ROUNDED_END_HOUR']
    return primeira.merge(ultima, on='DAY_HOUR', how='outer').fillna(0)
