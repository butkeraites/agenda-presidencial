import math
import pandas as pd
pd.options.plotting.backend = "plotly"
from sqlalchemy import create_engine
from datetime import date, datetime, timedelta
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

from nltk.corpus import stopwords
import nltk

complete_path = '/home/agendapresidencial/mysite/data/'
template='plotly_dark'

def get_callendar_types(df):
    df['MEETING_DATE'] = list(map(lambda x : pd.to_datetime(x, dayfirst=True).date(), df['MEETING_DATE']))
    return df

# LOADS
df_duracao_por_data = pd.read_csv(complete_path + 'df_duracao_por_data.csv')
df_duracao_por_dia_semana = pd.read_csv(complete_path + 'df_duracao_por_dia_semana.csv')
df_atividades_por_hora = pd.read_csv(complete_path + 'df_atividades_por_hora.csv')
df_atividades_por_local = pd.read_csv(complete_path + 'df_atividades_por_local.csv')
df_acumulado = pd.read_csv(complete_path + 'df_acumulado.csv')
completed_df = pd.read_csv(complete_path + 'completed_df.csv')
df_comeco_final_dia = pd.read_csv(complete_path + 'df_comeco_final_dia.csv')
df = get_callendar_types(pd.read_csv(complete_path + 'df.csv'))
dates_with_df = pd.read_csv(complete_path + 'dates_with_df.csv')
df_dia_uteis = pd.read_csv(complete_path + 'df_dia_uteis.csv')
df_dia_uteis_semana = pd.read_csv(complete_path + 'df_dia_uteis_semana.csv')

agrupado_agenda_mensal = px.bar(df_duracao_por_data,
                x="MES_REFERENCIA",
                y="MEETING_DURATION",
                barmode="group",
                labels={
                     "MES_REFERENCIA": "Mês de Referência",
                     "MEETING_DURATION": "Horas de Atividades Oficiais"
                 },
                 template=template)
agrupado_agenda_mensal.add_trace(go.Scatter(
    x=df_dia_uteis["MES_REFERENCIA"],
    y=df_dia_uteis["HORAS_DE_TRABALHO"],
    name="Horas CLT"
))


agrupado_agenda_semana = px.bar(df_duracao_por_dia_semana,
                x="DIA_DA_SEMANA",
                y="MEETING_DURATION",
                barmode="group",
                labels={
                     "DIA_DA_SEMANA": "Dia da Semana",
                     "MEETING_DURATION": "Horas de Atividades Oficiais"
                 },
                 template=template)

agrupado_agenda_semana.add_trace(go.Scatter(
    x=df_dia_uteis_semana["DIA_DA_SEMANA"],
    y=df_dia_uteis_semana["HORAS_DE_TRABALHO"],
    name="Horas CLT"
))


agrupado_agenda_hora_diaria = px.bar(df_atividades_por_hora,
                x="ROUNDED_BEGIN_HOUR",
                y="MEETING_ID",
                barmode="group",
                labels={
                     "ROUNDED_BEGIN_HOUR": "Hora de Inicio",
                     "MEETING_ID": "Qtd. de Atividades Oficias"
                 },
                 template=template)

agrupado_agenda_local = px.treemap(df_atividades_por_local,
                path=["MEETING_LOCATION"],
                values="MEETING_ID",
                labels={
                     "MEETING_LOCATION": "Local de realização da atividade",
                     "MEETING_ID": "Qtd. de Atividades Oficias"
                 },
                 template=template)

acumulado_diario = make_subplots(specs=[[{"secondary_y": True}]])
acumulado_diario.update_layout(template=template)
acumulado_diario.add_trace(go.Bar(
    x=df_acumulado["MEETING_DATE"],
    y=df_acumulado["CUM_MEETING_DURATION"],
    name="Presidência"
), secondary_y=False)

acumulado_diario.add_trace(go.Scatter(
    x=df_acumulado["MEETING_DATE"],
    y=df_acumulado["CUM_DIA_UTIL"],
    name="CLT"
), secondary_y=False)

acumulado_diario.add_trace(go.Scatter(
    x=df_acumulado["MEETING_DATE"],
    y=df_acumulado["CUM_MEETING_DURATION"]/df_acumulado["CUM_DIA_UTIL"],
    name="Comparativo"
), secondary_y=True)

# Set x-axis title
acumulado_diario.update_xaxes(title_text="Data")

# Set y-axes titles
acumulado_diario.update_yaxes(title_text="Acumulado Horas de Trabalho", secondary_y=False)
acumulado_diario.update_yaxes(title_text="Horas da Presidencia/CLT", secondary_y=True)

plot_por_similaridade = px.scatter(completed_df, x="DIMENSION_0", y="DIMENSION_1", color="MEETING_LOCATION",
                 size='MEETING_DURATION', hover_data=['MEETING_TITLE'], labels={
                     "DIMENSION_0" : "Dimensão de projeção x",
                     "DIMENSION_1" : "Dimensão de projeção y",
                     "MEETING_LOCATION" : "Local da atividade",
                     "MEETING_DURATION" : "Duração da atividade",
                     "MEETING_TITLE" : "Descrição da Atividade"
                 },
                 template=template)

primeira_ultima_atividade = make_subplots(specs=[[{"secondary_y": False}]])
primeira_ultima_atividade.update_layout(template=template)

primeira_ultima_atividade.add_trace(go.Bar(
    x=df_comeco_final_dia["DAY_HOUR"],
    y=df_comeco_final_dia["MEETING_DATE_x"],
    name="Horario Inicial"
), secondary_y=False)

primeira_ultima_atividade.add_trace(go.Bar(
    x=df_comeco_final_dia["DAY_HOUR"],
    y=df_comeco_final_dia["MEETING_DATE_y"],
    name="Horário Final"
), secondary_y=False)

# Set x-axis title
primeira_ultima_atividade.update_xaxes(title_text="Hora do Dia")
# Set y-axes titles
primeira_ultima_atividade.update_yaxes(title_text="Contagem de Dias", secondary_y=False)

#Média de horas trabalhadas por dia
media_geral_horas = df.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean()
media_geral = str(math.floor(media_geral_horas[0]))+'h'+str(math.floor((media_geral_horas[0] - math.floor(media_geral_horas[0]))*60))
media_ultimos_30_dias_horas = df[df['MEETING_DATE'] > (date.today()- timedelta(days=30))].groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean()
media_ultimos_30_dias = str(math.floor(media_ultimos_30_dias_horas[0]))+'h'+str(math.floor((media_ultimos_30_dias_horas[0] - math.floor(media_ultimos_30_dias_horas[0]))*60))

#Quantidade de dias úteis sem atividades oficiais
dates_with_df_null = dates_with_df[(dates_with_df['MEETING_DURATION'].isnull()) & (dates_with_df['DIA_UTIL'] == 1)]
dias_uteis_sem_atividades_oficiais = len(dates_with_df_null)

#Comparativo de trabalho do Presidente com CLT
comparativo_percentual = round(df_acumulado.tail(1)["CUM_MEETING_DURATION"]/df_acumulado.tail(1)["CUM_DIA_UTIL"]*100)
comparativo_percentual = str(int(list(comparativo_percentual)[0])) + "%"

# Initialise the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Define the app
app.layout = html.Div(className='row',
    children=[
        html.Div(className='three columns div-user-controls',
                    children=[
                        html.H2('Agenda Presidencial'),
                        dbc.Card([
                            dbc.CardHeader("Um típico dia de trabalho tem a duração de "),
                            dbc.CardBody(
                                [
                                    html.H4(media_geral, className="card-title"),
                                    html.P("desde a posse, ou", className="card-text"),
                                    html.H4(media_ultimos_30_dias, className="card-title"),
                                    html.P("nos ultimos 30 dias.", className="card-text"),
                                ]
                            )
                        ],
                        style={"width": "30rem"}),
                        dbc.Card([
                            dbc.CardHeader("Existem "),
                            dbc.CardBody(
                                [
                                    html.H4(str(dias_uteis_sem_atividades_oficiais) + " dias", className="card-title"),
                                    html.P("úteis sem atividades oficiais desde a posse", className="card-text")
                                ]
                            )
                        ],
                        style={"width": "30rem"}),
                        dbc.Card([
                            dbc.CardHeader("As atividades oficiais somam"),
                            dbc.CardBody(
                                [
                                    html.H4(comparativo_percentual, className="card-title"),
                                    html.P("da carga de trabalho de um funcionário CLT", className="card-text")
                                ]
                            )
                        ],
                        style={"width": "30rem"}),
                        html.P('Criado por @Renan_But'),
                        html.P('22-02-2021'),
                        html.P('agenda-presidencial@protonmail.com'),
                ]),
        html.Div(className='eight columns div-for-charts bg-grey',
                    children=[
                        html.Div(className='grafico-de-barra-pequeno', children = [
                            html.H2('Quantidade de atividades oficiais por hora de início'),
                            dcc.Graph(id='agrupado-agenda-diaria', figure = agrupado_agenda_hora_diaria)
                        ]),
                        html.Div(className='grafico-de-barra-pequeno', children = [
                            html.H2('Horário de início e fim dos dias com atividades oficiais'),
                            dcc.Graph(id='agrupado-agenda-diaria-unica', figure = primeira_ultima_atividade)
                        ]),
                        html.Div(className='grafico-de-barra-grande', children = [
                            html.H2('Horas de atividades oficiais por dia da semana'),
                            dcc.Graph(id='agrupado-agenda-semana', figure = agrupado_agenda_semana)
                        ]),
                        html.Div(className='grafico-de-barra-grande', children = [
                            html.H2('Horas de atividades oficiais por mês'),
                            dcc.Graph(id='agrupado-agenda-mensal', figure = agrupado_agenda_mensal)
                        ]),
                        html.Div(className='grafico-de-barra-grande', children = [
                            html.H2('Acumulado de horas de atividades oficiais'),
                            dcc.Graph(id='agrupado-diario', figure = acumulado_diario)
                        ]),
                        html.Div(className='diagrama-de-bloco', children = [
                            html.H2('Quantidade de atividades oficiais por local'),
                            dcc.Graph(id='agrupado-local', figure = agrupado_agenda_local)
                        ]),
                        html.Div(className='grafico-de-dispersao', children = [
                            html.H2('Atividades oficiais agrupadas por similaridade'),
                            dcc.Graph(id='agrupado-similaridade', figure = plot_por_similaridade)
                        ])
                    ])

], style={'width': '500'})


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)