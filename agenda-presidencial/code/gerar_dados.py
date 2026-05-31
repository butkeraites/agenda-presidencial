"""Orquestra o estágio de processamento.

Lê AGENDA_PRESIDENCIAL e CALENDARIO via repositorio, chama as agregações
puras de agregacoes.py e escreve os 11 CSVs consumidos por dashboard.py.

Uso: python gerar_dados.py
"""

import os
from datetime import date

import nltk
from nltk.corpus import stopwords

import agregacoes as ag
from calendario import calendario_para
from repositorio import RepositorioAgenda

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data') + os.sep
INICIO_DA_AGENDA = date(2019, 1, 1)


def stop_words_pt():
    nltk.download('stopwords', quiet=True)
    return list(stopwords.words('portuguese'))


def main(repo=None, hoje=None):
    if hoje is None:
        hoje = date.today()
    if repo is None:
        repo = RepositorioAgenda()

    compromissos = ag.compromissos_com_duracao(repo.carregar_compromissos())
    calendario = calendario_para(INICIO_DA_AGENDA, hoje)

    com_calendario = ag.compromissos_com_calendario(compromissos, calendario)
    duracao_por_mes = ag.duracao_por_mes(com_calendario)
    dia_uteis_mes = ag.dia_uteis_por_mes(calendario, ate_mes=duracao_por_mes['MES_REFERENCIA'].max())
    duracao_dia_semana = ag.duracao_por_dia_da_semana(com_calendario)
    dia_uteis_semana = ag.dia_uteis_por_dia_da_semana(calendario, ate_data=compromissos['MEETING_DATE'].max())
    por_hora = ag.atividades_por_hora(compromissos)
    por_local = ag.atividades_por_local(compromissos)
    acumulado, dates_with_df = ag.acumulado_diario(compromissos, calendario, hoje=hoje)
    com_projecao = ag.projecao_2d_titulos(compromissos, stop_words=stop_words_pt())
    comeco_final = ag.primeira_ultima_atividade_do_dia(compromissos)

    saidas = {
        'df_duracao_por_data.csv':       duracao_por_mes,
        'df_duracao_por_dia_semana.csv': duracao_dia_semana,
        'df_atividades_por_hora.csv':    por_hora,
        'df_atividades_por_local.csv':   por_local,
        'df_acumulado.csv':              acumulado,
        'completed_df.csv':              com_projecao,
        'df_comeco_final_dia.csv':       comeco_final,
        'df.csv':                        compromissos,
        'dates_with_df.csv':             dates_with_df,
        'df_dia_uteis.csv':              dia_uteis_mes,
        'df_dia_uteis_semana.csv':       dia_uteis_semana,
    }
    for nome, frame in saidas.items():
        frame.to_csv(DATA_DIR + nome)


if __name__ == '__main__':
    main()
