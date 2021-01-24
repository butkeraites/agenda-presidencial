import requests
import pandas as pd
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta


def get_all_dates(year, month, day):
    sdate = date(year, month, day)   # start date
    edate = date.today()   # end date
    delta = edate - sdate       # as timedelta
    all_dates = []
    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        all_dates.append(str(day))

    return all_dates

def prepare_calls(year, month, day):
    calls = {}
    all_dates = get_all_dates(year, month, day)
    for date in all_dates:
        url = 'https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/agenda-do-presidente-da-republica/' + date
        calls[date] = url
    return calls

def get_all_compromises_from_date(url, campos_de_interesse):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Compromissos do dia
    compromissos = []
    for campo in campos_de_interesse:
        elementos_campo = soup.find_all(class_=campo)
        for elemento_campo in elementos_campo:
            compromisso = {}
            for atributo in campos_de_interesse[campo]:
                valor_atributo = elemento_campo.find(class_=atributo)
                if valor_atributo:
                    compromisso[atributo] = valor_atributo.string
            compromissos.append(compromisso)
    return compromissos

def get_all_compromises(year, month, day, campos_de_interesse):
    calls = prepare_calls(year, month, day)
    compromises = {}
    for call in calls:
        print('Capturando ' + call + '[...]')
        compromises[call] = get_all_compromises_from_date(calls[call], campos_de_interesse)
    return compromises

def has_all_data(meeting, campos_de_interesse):
    all_data_found = True
    for data in campos_de_interesse['item-compromisso']:
        if data not in meeting:
            all_data_found = False
    return all_data_found

def transform_compromises_in_dataframe(year, month, day, initial_index):
    index = initial_index
    
    df_compromises = {
        'MEETING_ID' : [],
        'BEGIN_HOUR' : [],
        'END_HOUR' : [],
        'MEETING_TITLE' : [],
        'MEETING_LOCATION' : []
    }
    
    campos_de_interesse = {
        'item-compromisso' : [
            'compromisso-inicio',
            'compromisso-fim',
            'compromisso-titulo',
            'compromisso-local'
        ]
    }

    compromises = get_all_compromises(year, month, day, campos_de_interesse)
    
    for dates in compromises:
        print('Incluindo no DataFrame [' + dates + ']')
        for meeting in compromises[dates]:
            if has_all_data(meeting, campos_de_interesse):
                index += 1
                df_compromises['MEETING_ID'].append(index)
                df_compromises['BEGIN_HOUR'].append(datetime.strptime(dates + meeting['compromisso-inicio'], '%Y-%m-%d%Hh%M'))
                df_compromises['END_HOUR'].append(datetime.strptime(dates + meeting['compromisso-fim'], '%Y-%m-%d%Hh%M'))
                df_compromises['MEETING_TITLE'].append(meeting['compromisso-titulo'])
                df_compromises['MEETING_LOCATION'].append(meeting['compromisso-local'])
    
    return pd.DataFrame(df_compromises)

engine = create_engine('sqlite:////home/barbaruiva/Documents/Database/AGENDA_PRESIDENCIAL.db', echo=False)



# PRIMEIRA INCLUSAO DE REGISTROS NA BASE
#df_compromises = transform_compromises_in_dataframe(2019, 1, 1, 0)
#df_compromises.set_index('MEETING_ID', inplace=True)
#df_compromises.to_sql('AGENDA_PRESIDENCIAL', con=engine, if_exists='append')