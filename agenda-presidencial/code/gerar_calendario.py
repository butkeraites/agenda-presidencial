import os
import pandas as pd
from datetime import date, timedelta

import holidays

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, 'data', 'calendario.csv')

# Indexado por date.weekday(): 0 = segunda-feira ... 6 = domingo.
DIAS_DA_SEMANA = [
    'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira',
    'Sexta-feira', 'Sábado', 'Domingo'
]


def gerar_calendario(inicio, fim):
    feriados = holidays.Brazil(years=range(inicio.year, fim.year + 1))
    linhas = []
    dia = inicio
    while dia <= fim:
        eh_feriado = 1 if dia in feriados else 0
        eh_dia_util = 1 if dia.weekday() < 5 and not eh_feriado else 0
        linhas.append({
            'DATA': dia.strftime('%d/%m/%Y'),
            'MES_REFERENCIA': dia.replace(day=1).strftime('%d/%m/%Y'),
            'FERIADO': eh_feriado,
            'SEMANA_DO_ANO': dia.isocalendar()[1],
            'MES': dia.month,
            'ANO': dia.year,
            'DIA_DA_SEMANA': DIAS_DA_SEMANA[dia.weekday()],
            'DIA_UTIL': eh_dia_util,
        })
        dia += timedelta(days=1)
    return pd.DataFrame(linhas)


if __name__ == '__main__':
    df = gerar_calendario(date(2019, 1, 1), date.today())
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print('calendario.csv gerado com ' + str(len(df)) + ' linhas em ' + OUTPUT)
