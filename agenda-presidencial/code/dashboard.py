import pandas as pd
pd.options.plotting.backend = "plotly"
from sqlalchemy import create_engine
from datetime import date, datetime, timedelta
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

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

df = get_all_data()
df = get_meeting_date_and_duration(df)

df_callendar = get_all_callendar()
df_callendar = get_callendar_types(df_callendar)

df_with_dates = df.merge(df_callendar, on='MEETING_DATE', how='left')
df_duracao_por_data = df_with_dates.groupby('MES_REFERENCIA').sum()[['MEETING_DURATION']]
df_duracao_por_data = df_duracao_por_data.reset_index()

df_dia_uteis = df_callendar.groupby('MES_REFERENCIA').sum()[['DIA_UTIL']]
df_dia_uteis['HORAS_DE_TRABALHO'] = df_dia_uteis['DIA_UTIL'] * 8
df_dia_uteis = df_dia_uteis.reset_index()
df_dia_uteis = df_dia_uteis.loc[df_dia_uteis['MES_REFERENCIA'] <= max(df_duracao_por_data['MES_REFERENCIA'])] 

fig = px.bar(df_duracao_por_data,
                x="MES_REFERENCIA",
                y="MEETING_DURATION", 
                barmode="group",
                labels={
                     "MES_REFERENCIA": "Mês de Referência",
                     "MEETING_DURATION": "Horas de Atividades Oficiais"
                 })

fig.add_trace(go.Scatter(
    x=df_dia_uteis["MES_REFERENCIA"],
    y=df_dia_uteis["HORAS_DE_TRABALHO"],
    name="Horas CLT"
))


# Initialise the app
app = dash.Dash(__name__)

# Define the app
app.layout = html.Div(children=[
    html.H1(children='Soma das horas de atividades oficiais por mês:'),
    dcc.Graph(id='my-graph', figure = fig)
], style={'width': '500'})


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)