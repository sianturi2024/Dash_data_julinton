import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.graph_objs import Figure, Bar
import dash
from dash import dash, html, dcc 
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output, State
from dash import dash_table


df_weather = pd.read_csv('staging_focecast_hour.csv')

df = df_weather[['date', 'city', 'temp_c', 'humidity', 'wind_kph', 'condition_text']]

## average
average_weather = df.groupby('city').agg({
    'temp_c': 'mean',
    'wind_kph': 'mean',
    'humidity': 'mean'
}).reset_index()

average_weather

## name columns
column_renaming = {'city': 'city',  # Replace with desired names
                   'temp_c': 'avg_temp_c',
                   'wind_kph': 'avg_wind_kph',
                   'humidity': 'avg_humidity'}

# Rename the columns
average_weather.rename(columns=column_renaming, inplace=True)
average_weather

#figure
fig = px.choropleth(average_weather, 
                    locations='city', 
                    locationmode="country names",  
                    color='avg_humidity',
                    hover_name='city',
                    color_continuous_scale="Viridis",  
                    title='Average Weather by City'
                   )

# Show the plot
fig.show()

df_1 = pd.read_csv('location_target_weather.csv')

merged_df = average_weather.merge(df_1, how='left', on=['city'], sort=True)

#figure 1
fig1 = px.scatter_mapbox(
                        data_frame=merged_df,
                        lat='lat', 
                        lon='lon', 
                        hover_name='city', 
                        size='avg_temp_c',
                        color='city',
                        
                        # start location and zoom level
                        zoom=4, 
                        center={'lat': 51.1657, 'lon': 10.4515}, 
                        mapbox_style='carto-positron'
                       )
graph1 = dcc.Graph(figure=fig1)
fig1.show()

#fig2

fig2 = px.histogram(df, x="temp_c", color="city", marginal="box", title="Distribution of Temperature by City")
fig2.update_layout(bargap=0.1)  

graph2 = dcc.Graph(figure=fig2)

#fig3

fig3 = px.line(df, x="date", y="temp_c", color="city", title="Temperature Trend Over Time")

fig3.update_layout(xaxis_title="Date & Time")  

graph3 = dcc.Graph(figure=fig3)

#fig4
fig4 = px.box(df, x="condition_text", y="temp_c", title="Temperature Distribution by Weather Condition")

fig4.update_layout(xaxis_title="Weather Condition", yaxis_title="Temperature (°C)")
graph4 = dcc.Graph(figure=fig4)

#data cluster

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[['temp_c', 'humidity', 'wind_kph']]))

# Define number of clusters (adjust as needed)
n_clusters = 3

# K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(df_scaled)

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_

# Create a scatter plot with color-coding by cluster
fig5 = px.scatter(df, x="temp_c", y="humidity", color="cluster", title="Weather Data Clusters")

# Customize layout (adjust as needed)
fig5.update_layout(
    xaxis_title="Temperature (°C)",
    yaxis_title="Humidity (%)"
)

graph5 = dcc.Graph(figure=fig5)

#prediction temperature

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime 

df['date'] = pd.to_datetime(df['date'])  
df['day'] = df['date'].dt.day  #  can use dt.month after conversion

# Filter data for Berlin
berlin_data = df[df['city'] == 'Berlin']
X = berlin_data[['day']]
y = berlin_data['temp_c']

# Create and train a linear regression model for Berlin
model = LinearRegression()
model.fit(X, y)

# --- Define Prediction Range and Function ---

# Define the prediction date range (from 2023-12-20 to 2023-12-28)
start_date = pd.to_datetime('2023-12-20')
end_date = pd.to_datetime('2023-12-28')
prediction_dates = pd.date_range(start_date, end_date, inclusive='both')

# Function to make predictions for a date
def predict_temperature(date, model=model):
  day = date.day
  new_day = pd.DataFrame({'day': [day]})
  predicted_temperature = model.predict(new_day)[0]
  return predicted_temperature


# Filter actual temperatures for Berlin within the prediction range
berlin_actual_temps = berlin_data[
    (berlin_data['city'] == 'Berlin') &  
    (berlin_data['date'] >= start_date) &  
    (berlin_data['date'] <= end_date)  
]

print(f"\n**Predictions for Berlin from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:**")
for date in prediction_dates:
  predicted_temp = predict_temperature(date)
  actual_temp = berlin_actual_temps[berlin_actual_temps['date'] == date]['temp_c'].values[0]  # Access actual temperature
  print(f"{date.strftime('%Y-%m-%d')}: Predicted: {predicted_temp:.2f}°C, Actual: {actual_temp:.2f}°C")

  predicted_temperatures = []
actual_temperatures = []

# Loop through each prediction date and calculate predicted temperature
for date in prediction_dates:
    predicted_temp = predict_temperature(date)
    actual_temp = berlin_actual_temps[berlin_actual_temps['date'] == date]['temp_c'].values[0]
    predicted_temperatures.append(predicted_temp)
    actual_temperatures.append(actual_temp)

# Create a DataFrame to store predicted and actual temperatures
temperature_comparison_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted_Temperature': predicted_temperatures,
    'Actual_Temperature': actual_temperatures
})

# Create a line plot to compare predicted and actual temperatures
fig6 = px.line(temperature_comparison_df, x='Date', y=['Predicted_Temperature', 'Actual_Temperature'], 
              labels={'Date': 'Date', 'value': 'Temperature (°C)'}, 
              title='Predicted vs Actual Temperatures in Berlin from 2023-12-20 to 2023-12-20')

graph6 = dcc.Graph(figure=fig6)
fig6.show()

# Create a scatter plot with predicted and actual temperatures
fig7 = px.scatter(temperature_comparison_df, x='Date', 
                 y=['Predicted_Temperature', 'Actual_Temperature'], 
                 labels={'Date': 'Date', 'value': 'Temperature (°C)'}, 
                 title='Predicted vs Actual Temperatures in Berlin from 2023-12-20 to 2023-12-20',
                 color_discrete_map={'Predicted_Temperature': 'blue', 'Actual_Temperature': 'red'},
                 template='plotly_white'
                )

graph7 = dcc.Graph(figure=fig7)
fig7.show()

# paris

# Filter data for Paris
paris_data = df[df['city'] == 'Paris']
X = paris_data[['day']]
y = paris_data['temp_c']


model = LinearRegression()
model.fit(X, y)

# --- Define Prediction Range and Function ---

# Define the prediction date range (from 2023-12-20 to 2023-12-28)
start_date = pd.to_datetime('2023-12-20')
end_date = pd.to_datetime('2023-12-28')
prediction_dates = pd.date_range(start_date, end_date, inclusive='both')

# Function to make predictions for a date
def predict_temperature(date, model=model):
  day = date.day
  new_day = pd.DataFrame({'day': [day]})
  predicted_temperature = model.predict(new_day)[0]
  return predicted_temperature


# Filter actual temperatures for Berlin within the prediction range
paris_actual_temps = paris_data[
    (paris_data['city'] == 'Paris') &  
    (paris_data['date'] >= start_date) &  
    (paris_data['date'] <= end_date)  
]

print(f"\n**Predictions for Paris from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:**")
for date in prediction_dates:
  predicted_temp = predict_temperature(date)
  actual_temp = paris_actual_temps[paris_actual_temps['date'] == date]['temp_c'].values[0]  # Access actual temperature
  print(f"{date.strftime('%Y-%m-%d')}: Predicted: {predicted_temp:.2f}°C, Actual: {actual_temp:.2f}°C")

  predicted_temperatures = []
actual_temperatures = []

# Loop through each prediction date and calculate predicted temperature
for date in prediction_dates:
    predicted_temp = predict_temperature(date)
    actual_temp = paris_actual_temps[paris_actual_temps['date'] == date]['temp_c'].values[0]
    predicted_temperatures.append(predicted_temp)
    actual_temperatures.append(actual_temp)

# Create a DataFrame to store predicted and actual temperatures
temperature_comparison_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted_Temperature': predicted_temperatures,
    'Actual_Temperature': actual_temperatures
})

# Create a scatter plot with predicted and actual temperatures
fig8 = px.scatter(temperature_comparison_df, x='Date', 
                 y=['Predicted_Temperature', 'Actual_Temperature'], 
                 labels={'Date': 'Date', 'value': 'Temperature (°C)'}, 
                 title='Predicted vs Actual Temperatures in Paris from 2023-12-20 to 2023-12-20',
                 color_discrete_map={'Predicted_Temperature': 'blue', 'Actual_Temperature': 'red'},
                 template='plotly_white'
                )

graph8 = dcc.Graph(figure=fig8)
fig8.show()

# Djakarta

# Filter data for Paris
jakarta_data = df[df['city'] == 'Jakarta']
X = jakarta_data[['day']]
y = jakarta_data['temp_c']


model = LinearRegression()
model.fit(X, y)

# --- Define Prediction Range and Function ---

# Define the prediction date range (from 2023-12-20 to 2023-12-28)
start_date = pd.to_datetime('2023-12-20')
end_date = pd.to_datetime('2023-12-28')
prediction_dates = pd.date_range(start_date, end_date, inclusive='both')

# Function to make predictions for a date
def predict_temperature(date, model=model):
  day = date.day
  new_day = pd.DataFrame({'day': [day]})
  predicted_temperature = model.predict(new_day)[0]
  return predicted_temperature


# Filter actual temperatures for Berlin within the prediction range
jakarta_actual_temps = jakarta_data[
    (jakarta_data['city'] == 'Jakarta') &  
    (jakarta_data['date'] >= start_date) &  
    (jakarta_data['date'] <= end_date)  
]

print(f"\n**Predictions for Paris from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}:**")
for date in prediction_dates:
  predicted_temp = predict_temperature(date)
  actual_temp = jakarta_actual_temps[jakarta_actual_temps['date'] == date]['temp_c'].values[0]  # Access actual temperature
  print(f"{date.strftime('%Y-%m-%d')}: Predicted: {predicted_temp:.2f}°C, Actual: {actual_temp:.2f}°C")

  predicted_temperatures = []
actual_temperatures = []

# Loop through each prediction date and calculate predicted temperature
for date in prediction_dates:
    predicted_temp = predict_temperature(date)
    actual_temp = jakarta_actual_temps[jakarta_actual_temps['date'] == date]['temp_c'].values[0]
    predicted_temperatures.append(predicted_temp)
    actual_temperatures.append(actual_temp)

# Create a DataFrame to store predicted and actual temperatures
temperature_comparison_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted_Temperature': predicted_temperatures,
    'Actual_Temperature': actual_temperatures
})

# Create a scatter plot with predicted and actual temperatures
fig9 = px.scatter(temperature_comparison_df, x='Date', 
                 y=['Predicted_Temperature', 'Actual_Temperature'], 
                 labels={'Date': 'Date', 'value': 'Temperature (°C)'}, 
                 title='Predicted vs Actual Temperatures in Jakarta from 2023-12-20 to 2023-12-20',
                 color_discrete_map={'Predicted_Temperature': 'blue', 'Actual_Temperature': 'red'},
                 template='plotly_white'
                )

graph9 = dcc.Graph(figure=fig9)
fig9.show()

from dash import dash_table
import dash_bootstrap_components as dbc

d_table = dash_table.DataTable(df.to_dict('records'),
                                  [{"name": i, "id": i} for i in df.columns],
                               style_data={'color': 'white','backgroundColor': 'black'},
                               style_header={
                                  'backgroundColor': 'rgb(210, 210, 210)',
                                  'color': 'black','fontWeight': 'bold'
    })

# just adding the multi = True parameter for our dropdown

graph = dcc.Graph()
countries =df['city'].unique().tolist() 

app =dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

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
                                 d_table, graph1,  graph2, graph3, graph4, graph5, graph6, graph7, graph8, graph9])                      
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
    app.run_server()
