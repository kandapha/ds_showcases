import streamlit as st
import requests
import plotly.express as px


def get_data(city_name, n):
    KEY_API = '7cf0c0bc366645e4fe15677d6720a0f8'
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={KEY_API}"
    response = requests.get(url)
    data = response.json()
    filter_data = [data['list']]

    temps = []
    dates = []
    imgs = []
    for entry in filter_data[0][:n]:  # data[0] contains the list of weather records
        temp = entry["main"]["temp"] - 273.15
        #print('--',temp)
        temps.append(temp)
        date = entry["dt_txt"]
        dates.append(date)

        image = entry["weather"][0]["main"]
        imgs.append(image)
        
    return temps, dates, imgs

st.title("Weather Forcast fro the Next Day")
place = st.text_input(label="City Name (example : Tokyo, Osaka, Bangkok, Paris, Seoul)")
# Forecast Days slider
forecast_days = st.slider("Forecast Days", min_value=1, max_value=5, value=2)
n = forecast_days * 8
# Select box to choose the data type
data_type = st.selectbox("Select data to view", ("Temperature", "Sky"))
# Display the selection
st.subheader(f"{data_type} for the next {forecast_days} days in {place}")

if place:
    if data_type == 'Temperature':
        temps, dates, imgs = get_data(place, n)
        figure = px.line(x=dates, y=temps, labels={"x":"Date", "y":"Temperature(C)"})
        st.plotly_chart(figure)
    elif data_type == 'Sky':
        temps, dates, imgs = get_data(place, n)
        images= {"Clear":"images/clear.png", "Clouds":"images/cloud.png", "Rain":"images/rain.png", "Snow":"images/snow.png"}
        
        image_path = [images[img]  for img in imgs]
        st.image(image_path, caption=dates, width=100)