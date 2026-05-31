"""Calendário brasileiro: dia útil, feriado, semana do ano, mês de referência.

Função pura ``calendario_para(inicio, fim)`` devolve o DataFrame
diretamente, com tipos nativos. Sem CSV intermediário, sem tabela no
SQLite: o calendário é determinístico e barato de recomputar, então o
caller pede o intervalo que precisa.
"""

from datetime import timedelta

import holidays
import pandas as pd


# Indexado por date.weekday(): 0 = segunda-feira ... 6 = domingo.
DIAS_DA_SEMANA = [
    'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira',
    'Sexta-feira', 'Sábado', 'Domingo',
]


def calendario_para(inicio, fim):
    feriados = holidays.Brazil(years=range(inicio.year, fim.year + 1))
    linhas = []
    dia = inicio
    while dia <= fim:
        eh_feriado = 1 if dia in feriados else 0
        eh_dia_util = 1 if dia.weekday() < 5 and not eh_feriado else 0
        linhas.append({
            'MEETING_DATE': dia,
            'MES_REFERENCIA': dia.replace(day=1),
            'FERIADO': eh_feriado,
            'SEMANA_DO_ANO': dia.isocalendar()[1],
            'MES': dia.month,
            'ANO': dia.year,
            'DIA_DA_SEMANA': DIAS_DA_SEMANA[dia.weekday()],
            'DIA_UTIL': eh_dia_util,
        })
        dia += timedelta(days=1)
    return pd.DataFrame(linhas)
