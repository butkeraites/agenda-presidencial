"""Scraper da agenda presidencial.

Três peças com seam entre elas:

- ``fetch_dia(data)`` — adapter de rede. Único ponto que toca HTTP.
- ``parse_dia(html, data)`` — função pura. Recebe HTML e devolve
  ``list[dict]`` de compromissos. Testável com fixtures.
- ``coletar_intervalo(...)`` — orquestrador. ``fetch`` é injectável
  para que os testes substituam a rede por bytes locais.

Uso: ``python main.py`` (continua de onde parou via ``repositorio``).
"""

from datetime import date, datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

from repositorio import RepositorioAgenda

URL_BASE = 'https://www.gov.br/planalto/pt-br/acompanhe-o-planalto/agenda-do-presidente-da-republica/'

# gov.br rejeita o User-Agent padrao do requests; o timeout evita travar o backfill.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
}

# Estrutura HTML do site: contêiner -> atributos extraídos por classe CSS.
CAMPOS_DE_INTERESSE = {
    'item-compromisso': [
        'compromisso-inicio',
        'compromisso-fim',
        'compromisso-titulo',
        'compromisso-local',
    ]
}

COLUNAS = ['MEETING_ID', 'BEGIN_HOUR', 'END_HOUR', 'MEETING_TITLE', 'MEETING_LOCATION']


def url_para(data):
    return URL_BASE + data.isoformat()


def datas_no_intervalo(inicio, fim):
    return [inicio + timedelta(days=i) for i in range((fim - inicio).days + 1)]


def fetch_dia(data):
    """Adapter de rede: baixa o HTML do dia."""
    return requests.get(url_para(data), headers=HEADERS, timeout=30).content


def parse_dia(html, data, campos=CAMPOS_DE_INTERESSE):
    """Função pura: HTML + data -> lista de compromissos.

    Cada compromisso é um dict ``{BEGIN_HOUR, END_HOUR, MEETING_TITLE,
    MEETING_LOCATION}``. Compromissos sem todos os campos esperados são
    descartados silenciosamente (comportamento herdado).
    """
    soup = BeautifulSoup(html, 'html.parser')
    atributos = campos['item-compromisso']
    iso = data.isoformat()
    compromissos = []
    for elemento in soup.find_all(class_='item-compromisso'):
        raw = {}
        for atributo in atributos:
            tag = elemento.find(class_=atributo)
            if tag:
                raw[atributo] = tag.string
        if any(a not in raw for a in atributos):
            continue
        compromissos.append({
            'BEGIN_HOUR': datetime.strptime(iso + raw['compromisso-inicio'], '%Y-%m-%d%Hh%M'),
            'END_HOUR': datetime.strptime(iso + raw['compromisso-fim'], '%Y-%m-%d%Hh%M'),
            'MEETING_TITLE': raw['compromisso-titulo'],
            'MEETING_LOCATION': raw['compromisso-local'],
        })
    return compromissos


def coletar_intervalo(inicio, fim, proximo_id, fetch=fetch_dia):
    """Orquestra fetch + parse no intervalo ``[inicio, fim]``.

    ``proximo_id`` é o primeiro MEETING_ID a atribuir. ``fetch`` é
    injectável: passe uma função ``data -> bytes`` para testes.
    """
    linhas = []
    for data in datas_no_intervalo(inicio, fim):
        print('Capturando ' + data.isoformat() + '[...]')
        for compromisso in parse_dia(fetch(data), data):
            linhas.append({'MEETING_ID': proximo_id, **compromisso})
            proximo_id += 1
    return pd.DataFrame(linhas, columns=COLUNAS)


def main(repo=None, hoje=None):
    if repo is None:
        repo = RepositorioAgenda()
    if hoje is None:
        hoje = date.today()
    checkpoint = repo.checkpoint()
    if checkpoint is None:
        print('Base vazia: realizando carga inicial a partir de 2019-01-01')
        proximo_id, proxima_data = 1, date(2019, 1, 1)
    else:
        proximo_id, proxima_data = checkpoint
    df = coletar_intervalo(proxima_data, hoje, proximo_id)
    df.set_index('MEETING_ID', inplace=True)
    print('Numero de linhas a serem incluidas: ' + str(len(df)))
    repo.salvar_compromissos(df)


if __name__ == '__main__':
    main()
