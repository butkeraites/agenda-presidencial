import requests
from bs4 import BeautifulSoup

page = requests.get(
    'https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/agenda-do-presidente-da-republica/2021-01-22')
soup = BeautifulSoup(page.content, 'html.parser')

campos_de_interesse = {
    'item-compromisso' : [
        'compromisso-inicio',
        'compromisso-fim',
        'compromisso-titulo'
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

print(compromissos)