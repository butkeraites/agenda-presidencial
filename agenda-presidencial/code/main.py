import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta


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

def get_all_compromises_from_date(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')

    campos_de_interesse = {
        'item-compromisso' : [
            'compromisso-inicio',
            'compromisso-fim',
            'compromisso-titulo',
            'compromisso-local'
        ]
    }

    # Compromissos do dia
    compromissos = []
    for campo in campos_de_interesse:
        elementos_campo = soup.find_all(class_=campo)
        for elemento_campo in elementos_campo:
            compromisso = {}
            for atributo in campos_de_interesse[campo]:
                valor_atributo = elemento_campo.find(class_=atributo)
                compromisso[atributo] = valor_atributo.string
            compromissos.append(compromisso)
    return compromissos

def get_all_compromises(year, month, day):
    calls = prepare_calls(year, month, day)
    compromises = {}
    for call in calls:
        compromises[call] = get_all_compromises_from_date(calls[call])
    return compromises

print(get_all_compromises(2021,1,21))