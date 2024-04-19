import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.graph_objs import Figure, Bar
import dash
from dash import dash, html, dcc 
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output, State
from dash import dash_table

df = pd.read_csv('staging_focecast_hour.csv')

# just adding the multi = True parameter for our dropdown

graph = dcc.Graph()
countries =df['city'].unique().tolist() 

app =dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
server = app.server

#since we are using multi parameter, this time we need a list of the all unique values 
#in the "country" column to use in the function of the callback 

#changing the color of the dropdown value

dropdown = dcc.Dropdown(['Berlin', 'Osaka-Shi', 'Paris', 'London', 'Jakarta'], value=['Berlin', 'Osaka-Shi', 'Paris', 'London', 'Jakarta'], 
                        clearable=False, multi=True, style ={'paddingLeft': '30px', 
                                                             "backgroundColor": "#222222", "color": "#222222"})
#we added the styling to the dropdown menu

#we also moved the dropdown menu a bit to the left side

app.layout = html.Div([html.H1('Weather Data from 2023-04-11 to 2024-04-10', style={'textAlign': 'center', 'color': '#636EFA'}), 
                       html.Div(html.P("Weather Analysis of Berlin, Osaka-Shi, Paris, London, Jakarta"), 
                                style={'marginLeft': 50, 'marginRight': 25}),
                       html.Div([html.Div('Weather Data', 
                                          style={'backgroundColor': '#636EFA', 'color': 'white', 
                                                 'width': '900px', 'marginLeft': 'auto', 'marginRight': 'auto'}),
                                 d_table, graph1,  graph2, graph3, graph4, graph5, graph6])                      
                      ])
@callback(
    Output(graph1, "figure"), 
    Input(dropdown, "value")) # we did not give it to a id=... that's why we can just put the value name

def update_bar_chart(cities): 
    mask = df["city"].isin(cities) # coming from the function parameter
    fig =px.bar(df[mask], 
             x='date', 
             y='temp_c',  
             color='city',
             barmode='group',
             height=300, title = "Temperature Trend Over Time",)
    fig = fig.update_layout(
        plot_bgcolor="#222222", paper_bgcolor="#222222", font_color="white"
    )

    return fig # whatever you are returning here is connected to the component property of
                       #the output which is figure

if __name__ == "__main__":
    app.run_server(port=8081)