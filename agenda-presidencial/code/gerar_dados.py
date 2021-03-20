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

complete_path = '/home/barbaruiva/Documents/agenda-presidencial/agenda-presidencial/code/data/'

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

#Agrupado atividades oficiais por hora de inicio
df_atividades_por_hora = df.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_ID'].count().reset_index()

#Agrupado atividades oficiais por local
df_atividades_por_local = df.groupby(['MEETING_LOCATION'])['MEETING_ID'].count().reset_index()

#Acumulado diario de horas de atividades oficiais
df_diario = df.groupby('MEETING_DATE').sum()[['MEETING_DURATION']].reset_index()
dates_with_df = df_callendar.merge(df_diario, on='MEETING_DATE', how='left')
dates_with_df = dates_with_df[(dates_with_df['MEETING_DATE'] <= date.today())]
df_acumulado = dates_with_df[['MEETING_DATE', 'MEETING_DURATION', 'DIA_UTIL']].fillna(0)
df_acumulado = pd.concat([df_acumulado, df_acumulado[['MEETING_DURATION', 'DIA_UTIL']].cumsum().add_prefix('CUM_')],axis=1)
df_acumulado['CUM_DIA_UTIL'] = df_acumulado['CUM_DIA_UTIL'] * 8

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

#Primeiro compromisso dos dias
df_primeira_atividade_do_dia = df.groupby('MEETING_DATE').agg({'ROUNDED_BEGIN_HOUR': 'min'}).reset_index()
df_primeira_atividade_do_dia = df_primeira_atividade_do_dia.groupby(['ROUNDED_BEGIN_HOUR'])['MEETING_DATE'].count().reset_index()
df_ultima_atividade_do_dia = df.groupby('MEETING_DATE').agg({'ROUNDED_END_HOUR': 'max'}).reset_index()
df_ultima_atividade_do_dia = df_ultima_atividade_do_dia.groupby(['ROUNDED_END_HOUR'])['MEETING_DATE'].count().reset_index()
df_primeira_atividade_do_dia['DAY_HOUR'] = df_primeira_atividade_do_dia['ROUNDED_BEGIN_HOUR']
df_ultima_atividade_do_dia['DAY_HOUR'] = df_ultima_atividade_do_dia['ROUNDED_END_HOUR']

df_comeco_final_dia = df_primeira_atividade_do_dia.merge(df_ultima_atividade_do_dia, on='DAY_HOUR', how='outer').fillna(0)


df_duracao_por_data.to_csv(complete_path + 'df_duracao_por_data.csv')
df_duracao_por_dia_semana.to_csv(complete_path + 'df_duracao_por_dia_semana.csv')
df_atividades_por_hora.to_csv(complete_path + 'df_atividades_por_hora.csv')
df_atividades_por_local.to_csv(complete_path + 'df_atividades_por_local.csv')
df_acumulado.to_csv(complete_path + 'df_acumulado.csv')
completed_df.to_csv(complete_path + 'completed_df.csv')
df_comeco_final_dia.to_csv(complete_path + 'df_comeco_final_dia.csv')
df.to_csv(complete_path + 'df.csv')
dates_with_df.to_csv(complete_path + 'dates_with_df.csv')
df_dia_uteis.to_csv(complete_path + 'df_dia_uteis.csv')
df_dia_uteis_semana.to_csv(complete_path + 'df_dia_uteis_semana.csv')