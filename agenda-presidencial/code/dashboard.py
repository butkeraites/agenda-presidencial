import math
import os
from datetime import date

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dados_do_dashboard import (
    carregar,
    comparativo_clt_percentual,
    dias_uteis_sem_atividades,
    media_horas_por_dia,
    media_horas_ultimos_30_dias,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
TEMPLATE = 'plotly_dark'


def formato_hm(horas):
    h = math.floor(horas)
    return f'{h}h{math.floor((horas - h) * 60)}'


dados = carregar(DATA_DIR)
hoje = date.today()

# ---------------- figuras ----------------

agrupado_agenda_mensal = px.bar(
    dados.duracao_por_mes,
    x='MES_REFERENCIA', y='MEETING_DURATION', barmode='group',
    labels={'MES_REFERENCIA': 'Mês de Referência', 'MEETING_DURATION': 'Horas de Atividades Oficiais'},
    template=TEMPLATE,
)
agrupado_agenda_mensal.add_trace(go.Scatter(
    x=dados.dia_uteis_por_mes['MES_REFERENCIA'],
    y=dados.dia_uteis_por_mes['HORAS_DE_TRABALHO'],
    name='Horas CLT',
))

agrupado_agenda_semana = px.bar(
    dados.duracao_por_dia_da_semana,
    x='DIA_DA_SEMANA', y='MEETING_DURATION', barmode='group',
    labels={'DIA_DA_SEMANA': 'Dia da Semana', 'MEETING_DURATION': 'Horas de Atividades Oficiais'},
    template=TEMPLATE,
)
agrupado_agenda_semana.add_trace(go.Scatter(
    x=dados.dia_uteis_por_dia_da_semana['DIA_DA_SEMANA'],
    y=dados.dia_uteis_por_dia_da_semana['HORAS_DE_TRABALHO'],
    name='Horas CLT',
))

agrupado_agenda_hora_diaria = px.bar(
    dados.atividades_por_hora,
    x='ROUNDED_BEGIN_HOUR', y='MEETING_ID', barmode='group',
    labels={'ROUNDED_BEGIN_HOUR': 'Hora de Inicio', 'MEETING_ID': 'Qtd. de Atividades Oficias'},
    template=TEMPLATE,
)

agrupado_agenda_local = px.treemap(
    dados.atividades_por_local,
    path=['MEETING_LOCATION'], values='MEETING_ID',
    labels={'MEETING_LOCATION': 'Local de realização da atividade', 'MEETING_ID': 'Qtd. de Atividades Oficias'},
    template=TEMPLATE,
)

fig_acumulado_diario = make_subplots(specs=[[{'secondary_y': True}]])
fig_acumulado_diario.update_layout(template=TEMPLATE)
fig_acumulado_diario.add_trace(go.Bar(
    x=dados.acumulado_diario['MEETING_DATE'],
    y=dados.acumulado_diario['CUM_MEETING_DURATION'],
    name='Presidência',
), secondary_y=False)
fig_acumulado_diario.add_trace(go.Scatter(
    x=dados.acumulado_diario['MEETING_DATE'],
    y=dados.acumulado_diario['CUM_DIA_UTIL'],
    name='CLT',
), secondary_y=False)
fig_acumulado_diario.add_trace(go.Scatter(
    x=dados.acumulado_diario['MEETING_DATE'],
    y=dados.acumulado_diario['CUM_MEETING_DURATION'] / dados.acumulado_diario['CUM_DIA_UTIL'],
    name='Comparativo',
), secondary_y=True)
fig_acumulado_diario.update_xaxes(title_text='Data')
fig_acumulado_diario.update_yaxes(title_text='Acumulado Horas de Trabalho', secondary_y=False)
fig_acumulado_diario.update_yaxes(title_text='Horas da Presidencia/CLT', secondary_y=True)

plot_por_similaridade = px.scatter(
    dados.compromissos_com_projecao,
    x='DIMENSION_0', y='DIMENSION_1',
    color='MEETING_LOCATION', size='MEETING_DURATION',
    hover_data=['MEETING_TITLE'],
    labels={
        'DIMENSION_0': 'Dimensão de projeção x',
        'DIMENSION_1': 'Dimensão de projeção y',
        'MEETING_LOCATION': 'Local da atividade',
        'MEETING_DURATION': 'Duração da atividade',
        'MEETING_TITLE': 'Descrição da Atividade',
    },
    template=TEMPLATE,
)

primeira_ultima_atividade = make_subplots(specs=[[{'secondary_y': False}]])
primeira_ultima_atividade.update_layout(template=TEMPLATE)
primeira_ultima_atividade.add_trace(go.Bar(
    x=dados.comeco_final_do_dia['DAY_HOUR'],
    y=dados.comeco_final_do_dia['MEETING_DATE_x'],
    name='Horario Inicial',
), secondary_y=False)
primeira_ultima_atividade.add_trace(go.Bar(
    x=dados.comeco_final_do_dia['DAY_HOUR'],
    y=dados.comeco_final_do_dia['MEETING_DATE_y'],
    name='Horário Final',
), secondary_y=False)
primeira_ultima_atividade.update_xaxes(title_text='Hora do Dia')
primeira_ultima_atividade.update_yaxes(title_text='Contagem de Dias', secondary_y=False)

# ---------------- KPIs ----------------

kpi_media_geral          = formato_hm(media_horas_por_dia(dados))
kpi_media_ultimos_30     = formato_hm(media_horas_ultimos_30_dias(dados, hoje))
kpi_dias_sem_atividades  = str(dias_uteis_sem_atividades(dados)) + ' dias'
kpi_comparativo_clt      = str(comparativo_clt_percentual(dados)) + '%'

# ---------------- app ----------------

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div(className='row', children=[
    html.Div(className='three columns div-user-controls', children=[
        html.H2('Agenda Presidencial'),
        dbc.Card([
            dbc.CardHeader('Um típico dia de trabalho tem a duração de '),
            dbc.CardBody([
                html.H4(kpi_media_geral, className='card-title'),
                html.P('desde a posse, ou', className='card-text'),
                html.H4(kpi_media_ultimos_30, className='card-title'),
                html.P('nos ultimos 30 dias.', className='card-text'),
            ]),
        ], style={'width': '30rem'}),
        dbc.Card([
            dbc.CardHeader('Existem '),
            dbc.CardBody([
                html.H4(kpi_dias_sem_atividades, className='card-title'),
                html.P('úteis sem atividades oficiais desde a posse', className='card-text'),
            ]),
        ], style={'width': '30rem'}),
        dbc.Card([
            dbc.CardHeader('As atividades oficiais somam'),
            dbc.CardBody([
                html.H4(kpi_comparativo_clt, className='card-title'),
                html.P('da carga de trabalho de um funcionário CLT', className='card-text'),
            ]),
        ], style={'width': '30rem'}),
        html.P('Criado por @Renan_But'),
        html.P('22-02-2021'),
        html.P('agenda-presidencial@protonmail.com'),
    ]),
    html.Div(className='eight columns div-for-charts bg-grey', children=[
        html.Div(className='grafico-de-barra-pequeno', children=[
            html.H2('Quantidade de atividades oficiais por hora de início'),
            dcc.Graph(id='agrupado-agenda-diaria', figure=agrupado_agenda_hora_diaria),
        ]),
        html.Div(className='grafico-de-barra-pequeno', children=[
            html.H2('Horário de início e fim dos dias com atividades oficiais'),
            dcc.Graph(id='agrupado-agenda-diaria-unica', figure=primeira_ultima_atividade),
        ]),
        html.Div(className='grafico-de-barra-grande', children=[
            html.H2('Horas de atividades oficiais por dia da semana'),
            dcc.Graph(id='agrupado-agenda-semana', figure=agrupado_agenda_semana),
        ]),
        html.Div(className='grafico-de-barra-grande', children=[
            html.H2('Horas de atividades oficiais por mês'),
            dcc.Graph(id='agrupado-agenda-mensal', figure=agrupado_agenda_mensal),
        ]),
        html.Div(className='grafico-de-barra-grande', children=[
            html.H2('Acumulado de horas de atividades oficiais'),
            dcc.Graph(id='agrupado-diario', figure=fig_acumulado_diario),
        ]),
        html.Div(className='diagrama-de-bloco', children=[
            html.H2('Quantidade de atividades oficiais por local'),
            dcc.Graph(id='agrupado-local', figure=agrupado_agenda_local),
        ]),
        html.Div(className='grafico-de-dispersao', children=[
            html.H2('Atividades oficiais agrupadas por similaridade'),
            dcc.Graph(id='agrupado-similaridade', figure=plot_por_similaridade),
        ]),
    ]),
], style={'width': '500'})


if __name__ == '__main__':
    app.run_server(debug=True)
