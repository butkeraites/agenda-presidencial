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

#Média de horas trabalhadas por dia
media_geral_horas = df.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean()
media_geral = str(math.floor(media_geral_horas[0]))+'h'+str(math.floor((media_geral_horas[0] - math.floor(media_geral_horas[0]))*60))
media_ultimos_30_dias_horas = df[df['MEETING_DATE'] > (date.today()- timedelta(days=30))].groupby('MEETING_DATE').sum()[['MEETING_DURATION']].mean()
media_ultimos_30_dias = str(math.floor(media_ultimos_30_dias_horas[0]))+'h'+str(math.floor((media_ultimos_30_dias_horas[0] - math.floor(media_ultimos_30_dias_horas[0]))*60))

# Initialise the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Define the app
app.layout = html.Div(className='row',
    children=[
        html.Div(className='two columns div-user-controls',
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
                        style={"width": "30rem"})
                ]),
        html.Div(className='eight columns div-for-charts bg-grey',
                    children=[
                        html.Div(className='grafico-de-barra-pequeno', children = [
                            html.H2('Quantidade de atividades oficiais por hora de início'),
                            dcc.Graph(id='agrupado-agenda-diaria', figure = agrupado_agenda_hora_diaria)
                        ]),
                        html.Div(className='grafico-de-barra-grande', children = [
                            html.H2('Horas de atividades oficiais por mês'),
                            dcc.Graph(id='agrupado-agenda-mensal', figure = agrupado_agenda_mensal)
                        ]),
                        html.Div(className='grafico-de-barra-grande', children = [
                            html.H2('Horas de atividades oficiais por dia da semana'),
                            dcc.Graph(id='agrupado-agenda-semana', figure = agrupado_agenda_semana)
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