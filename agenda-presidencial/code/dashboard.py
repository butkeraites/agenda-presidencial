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

template='plotly_dark'

def return_stop_words_portuguese():
    words = []
    nltk.download('stopwords')
    language = "portuguese"
    for word in stopwords.words(language):
        words.append(word)
    return words


def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    t_rounded = t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30)
    return (t_rounded.hour)


def get_all_data():
    engine = create_engine('sqlite:////home/barbaruiva/Documents/Database/AGENDA_PRESIDENCIAL.db', echo=False)
    conn = engine.connect()
    return pd.read_sql_query("SELECT * FROM AGENDA_PRESIDENCIAL", engine)

def get_all_callendar():
    engine = create_engine('sqlite:////home/barbaruiva/Documents/Database/AGENDA_PRESIDENCIAL.db', echo=False)
    conn = engine.connect()
    return pd.read_sql_query("SELECT * FROM CALENDARIO", engine)

def get_meeting_date_and_duration(df):
    df['BEGIN_HOUR'] = pd.to_datetime(df['BEGIN_HOUR'])
    df['ROUNDED_BEGIN_HOUR'] = list(map(lambda x : hour_rounder(x), df['BEGIN_HOUR']))
    df['END_HOUR'] = pd.to_datetime(df['END_HOUR'])
    df['ROUNDED_END_HOUR'] = list(map(lambda x : hour_rounder(x), df['END_HOUR']))
    df['MEETING_DATE'] = list(map(lambda x : x.date(), df['BEGIN_HOUR']))
    df['MEETING_DURATION'] = list(map(lambda x : x.total_seconds()/3600,df['END_HOUR'] - df['BEGIN_HOUR']))
    return df

def get_callendar_types(df):
    df['MEETING_DATE'] = pd.to_datetime(df['DATA'], dayfirst=True)
    df['MEETING_DATE'] = list(map(lambda x : x.date(), df['MEETING_DATE']))
    df['MES_REFERENCIA'] = pd.to_datetime(df['MES_REFERENCIA'], dayfirst=True)
    df['MES_REFERENCIA'] = list(map(lambda x : x.date(), df['MES_REFERENCIA']))
    df['FERIADO'] = pd.to_numeric(df['FERIADO'])
    df['SEMANA_DO_ANO'] = pd.to_numeric(df['SEMANA_DO_ANO'])
    df['MES'] = pd.to_numeric(df['MES'])
    df['ANO'] = pd.to_numeric(df['ANO'])
    return df

#Agenda presidencial
df = get_all_data()
df = get_meeting_date_and_duration(df)

#Calendario
df_callendar = get_all_callendar()
df_callendar = get_callendar_types(df_callendar)

#Agrupado de horas de atividades oficiais por mes
df_with_dates = df.merge(df_callendar, on='MEETING_DATE', how='left')
df_duracao_por_data = df_with_dates.groupby('MES_REFERENCIA').sum()[['MEETING_DURATION']].reset_index()

#Horas de trabalho de uma pessoa com CLT
df_dia_uteis = df_callendar.groupby('MES_REFERENCIA').sum()[['DIA_UTIL']]
df_dia_uteis['HORAS_DE_TRABALHO'] = df_dia_uteis['DIA_UTIL'] * 8
df_dia_uteis = df_dia_uteis.reset_index()
df_dia_uteis = df_dia_uteis.loc[df_dia_uteis['MES_REFERENCIA'] <= max(df_duracao_por_data['MES_REFERENCIA'])] 

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

#Agrupado de horas de atividades oficiais por dia da semana
ordem_dia_semana = ["Domingo","Segunda-feira","Terça-feira","Quarta-feira","Quinta-feira","Sexta-feira","Sábado"]
df_duracao_por_dia_semana = df_with_dates.groupby('DIA_DA_SEMANA').sum()[['MEETING_DURATION']].reset_index()
df_duracao_por_dia_semana['DIA_DA_SEMANA'] = pd.CategoricalIndex(df_duracao_por_dia_semana['DIA_DA_SEMANA'], ordered=True, categories=ordem_dia_semana)
df_duracao_por_dia_semana = df_duracao_por_dia_semana.sort_values('DIA_DA_SEMANA')

#Horas de trabalho de uma pessoa com CLT
df_dia_uteis_semana = df_callendar.loc[df_callendar['MEETING_DATE'] <= max(df_with_dates['MEETING_DATE'])].groupby('DIA_DA_SEMANA').sum()[['DIA_UTIL']]
df_dia_uteis_semana['HORAS_DE_TRABALHO'] = df_dia_uteis_semana['DIA_UTIL'] * 8
df_dia_uteis_semana = df_dia_uteis_semana.reset_index()
df_dia_uteis_semana['DIA_DA_SEMANA'] = pd.CategoricalIndex(df_dia_uteis_semana['DIA_DA_SEMANA'], ordered=True, categories=ordem_dia_semana)
df_dia_uteis_semana = df_dia_uteis_semana.sort_values('DIA_DA_SEMANA')

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

#Agrupado atividades oficiais por hora de inicio
df_atividades_por_hora = df.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_ID'].count().reset_index()
agrupado_agenda_hora_diaria = px.bar(df_atividades_por_hora,
                x="ROUNDED_BEGIN_HOUR",
                y="MEETING_ID", 
                barmode="group",
                labels={
                     "ROUNDED_BEGIN_HOUR": "Hora de Inicio",
                     "MEETING_ID": "Qtd. de Atividades Oficias"
                 }, 
                 template=template)


#Agrupado atividades oficiais por local
df_atividades_por_local = df.groupby(['MEETING_LOCATION'])['MEETING_ID'].count().reset_index()
agrupado_agenda_local = px.treemap(df_atividades_por_local,
                path=["MEETING_LOCATION"],
                values="MEETING_ID", 
                labels={
                     "MEETING_LOCATION": "Local de realização da atividade",
                     "MEETING_ID": "Qtd. de Atividades Oficias"
                 }, 
                 template=template)

#Acumulado diario de horas de atividades oficiais
df_diario = df.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].reset_index()
dates_with_df = df_callendar.merge(df_diario, on='MEETING_DATE', how='left')
dates_with_df = dates_with_df[(dates_with_df['MEETING_DATE'] <= date.today())]
df_acumulado = dates_with_df[['MEETING_DATE', 'MEETING_DURATION', 'DIA_UTIL']].fillna(0)
df_acumulado = pd.concat([df_acumulado, df_acumulado[['MEETING_DURATION', 'DIA_UTIL']].cumsum().add_prefix('CUM_')],axis=1)
df_acumulado['CUM_DIA_UTIL'] = df_acumulado['CUM_DIA_UTIL'] * 8

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

#Examinando os tipos de reuniões que foram realizadas
tf_idf_vec_smooth = TfidfVectorizer(use_idf=True,  
                        smooth_idf=True,  
                        ngram_range=(1,4),stop_words=return_stop_words_portuguese())
tf_idf_data_smooth = tf_idf_vec_smooth.fit_transform(list(df['MEETING_TITLE']))
tf_idf_dataframe_smooth=pd.DataFrame(tf_idf_data_smooth.toarray(),columns=tf_idf_vec_smooth.get_feature_names())

pca = PCA(n_components = 2)
df_reduced_dim = pd.DataFrame(data=pca.fit_transform(tf_idf_dataframe_smooth))
df_reduced_dim = df_reduced_dim.add_prefix('DIMENSION_').reset_index()
completed_df = df.reset_index().merge(df_reduced_dim, on='index', how='left').drop(['index'], axis=1)

plot_por_similaridade = px.scatter(completed_df, x="DIMENSION_0", y="DIMENSION_1", color="MEETING_LOCATION",
                 size='MEETING_DURATION', hover_data=['MEETING_TITLE'], labels={
                     "DIMENSION_0" : "Dimensão de projeção x",
                     "DIMENSION_1" : "Dimensão de projeção y",
                     "MEETING_LOCATION" : "Local da atividade",
                     "MEETING_DURATION" : "Duração da atividade",
                     "MEETING_TITLE" : "Descrição da Atividade"
                 }, 
                 template=template)

#Primeiro compromisso dos dias
df_primeira_atividade_do_dia = df.groupby('MEETING_DATE').agg({'ROUNDED_BEGIN_HOUR': 'min'}).reset_index()
df_primeira_atividade_do_dia = df_primeira_atividade_do_dia.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_DATE'].count().reset_index()
df_ultima_atividade_do_dia = df.groupby('MEETING_DATE').agg({'ROUNDED_END_HOUR': 'max'}).reset_index()
df_ultima_atividade_do_dia = df_ultima_atividade_do_dia.groupby(['ROUNDED_END_HOUR'])['MEETING_DATE'].count().reset_index()
df_primeira_atividade_do_dia['DAY_HOUR'] = df_primeira_atividade_do_dia['ROUNDED_BEGIN_HOUR']
df_ultima_atividade_do_dia['DAY_HOUR'] = df_ultima_atividade_do_dia['ROUNDED_END_HOUR']

df_comeco_final_dia = df_primeira_atividade_do_dia.merge(df_ultima_atividade_do_dia, on='DAY_HOUR', how='outer').fillna(0)

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
                        style={"width": "30rem"})
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