import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMapWithTime, TimestampedGeoJson

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from folium.raster_layers import TileLayer

tweets = pd.read_csv(r'C:\Users\vedan\Downloads\InfoVizData\covid19_tweets.csv')
cities = pd.read_csv(r'C:\Users\vedan\Downloads\InfoVizData\worldcities.csv')

tweets["lat"] = np.NAN
tweets["lng"] = np.NAN
tweets.rename({'user_location': 'location'}, axis=1, inplace=True)

location_list = tweets['location'].fillna(value='').str.split(',')

myset1 = set(cities['city'].fillna(value = '').str.lower().str.strip().values.tolist())
myset2 = set(cities['admin_name'].str.lower().str.strip().values.tolist())
myset3 = set(cities['country'].str.lower().str.strip().values.tolist())
myset4 = set(cities['iso2'].str.lower().str.strip().values.tolist())
myset5 = set(cities['iso3'].str.lower().str.strip().values.tolist())


latitudes = cities['lat'].fillna(value = '').values.tolist()
longitudes = cities['lng'].fillna(value = '').values.tolist()


for i in range(len(location_list)):
    for loc in location_list[i]:
        loc = loc.lower().strip()
        if loc in list(myset1):
            tweets['lat'][i] = latitudes[list(myset1).index(loc)]
            tweets['lng'][i] = longitudes[list(myset1).index(loc)]
        elif loc in list(myset2):
            tweets['lat'][i] = latitudes[list(myset2).index(loc)]
            tweets['lng'][i] = longitudes[list(myset2).index(loc)]
        elif loc in list(myset3):
            tweets['lat'][i] = latitudes[list(myset3).index(loc)]
            tweets['lng'][i] = longitudes[list(myset3).index(loc)]
        elif loc in list(myset4):
            tweets['lat'][i] = latitudes[list(myset4).index(loc)]
            tweets['lng'][i] = longitudes[list(myset4).index(loc)]
        elif loc in list(myset5):
            tweets['lat'][i] = latitudes[list(myset5).index(loc)]
            tweets['lng'][i] = longitudes[list(myset5).index(loc)]

tweets['clean_text'] = tweets['text']
tweets['date_orig'] = tweets['date']

tweets['lat'] = tweets['lat'].astype(float)
tweets['lng'] = tweets['lng'].astype(float)
tweets['date'] = tweets['date'].str.split(' ').str.get(0)


tweets_geo = tweets[['lat','lng','location','date','date_orig']].dropna()

tweets_geo = tweets_geo[tweets_geo['lat'] >= 19.50139]
tweets_geo = tweets_geo[tweets_geo['lat'] <= 64.85694]
tweets_geo = tweets_geo[tweets_geo['lng'] >= -161.75583]
tweets_geo = tweets_geo[tweets_geo['lng'] <= -68.01197]

lat_mean = tweets_geo['lat'].mean()
lng_mean = tweets_geo['lng'].mean()

sw = tweets_geo[['lat', 'lng']].min().values.tolist()
ne = tweets_geo[['lat', 'lng']].max().values.tolist()

daywise_tweets = folium.Map(location=[lat_mean,lng_mean], tiles=None, min_zoom=2) 
daywise_tweets.fit_bounds([sw, ne]) 

TileLayer(
    tiles='https://map1.vis.earthdata.nasa.gov/wmts-webmerc/{variant}/default/{time}/{tilematrixset}{maxZoom}/{z}/{y}/{x}.{format}',
    attr='Imagery provided by services from the Global Imagery Browse Services (GIBS), operated by the NASA/GSFC/Earth Science Data and Information System ' +'(<a href="https://earthdata.nasa.gov">ESDIS</a>)',
    bounds= [sw, ne],
    maxZoom= 8,
    format= 'jpg',
    time= '',
    tilematrixset= 'GoogleMapsCompatible_Level',
    variant= 'VIIRS_CityLights_2012'
).add_to(daywise_tweets)

tweets_map = []
for date in tweets_geo['date'].str.split(' ').str.get(0).unique().tolist():
    temp = []
    for index, row in tweets_geo[tweets_geo['date'] == date].iterrows():
        temp.append([row['lat'],row['lng']])
    tweets_map.append(temp)

heat = HeatMapWithTime(data=tweets_map, name=None, radius=4, min_opacity=0, max_opacity=0.5, gradient={0.5:'green', 1:'red'}, auto_play=False, index_steps=1,
                    speed_step=0.1, position='bottomleft', overlay=True, control=True, show=True)

heat.add_to(daywise_tweets)

daywise_tweets          #Map1



timely_tweets = folium.Map(location=[lat_mean,lng_mean], tiles=None, min_zoom=2) 
timely_tweets.fit_bounds([sw, ne]) 

TileLayer(
    tiles='https://map1.vis.earthdata.nasa.gov/wmts-webmerc/{variant}/default/{time}/{tilematrixset}{maxZoom}/{z}/{y}/{x}.{format}',
    attr='Imagery provided by services from the Global Imagery Browse Services (GIBS), operated by the NASA/GSFC/Earth Science Data and Information System ' +'(<a href="https://earthdata.nasa.gov">ESDIS</a>)',
    bounds= [sw, ne],
    #minZoom= 1,
    maxZoom= 8,
    format= 'jpg',
    time= '',
    tilematrixset= 'GoogleMapsCompatible_Level',
    variant= 'VIIRS_CityLights_2012'
).add_to(timely_tweets)

features2 = []
for _, row in tweets_geo.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': { 'type':'Point','coordinates':[row['lng'],row['lat']]
            },
            'properties': {
                'iconstyle':{'fillOpacity': 0, 'radius': 1},
                'time': row['date'],
                'icon': 'circle'
            }
        }
        features2.append(feature)

m = TimestampedGeoJson(features2, transition_time=500, auto_play=False,
                       add_last_point=True, period='P1D', duration=None).add_to(timely_tweets)
timely_tweets.fit_bounds([sw, ne]) 

timely_tweets          #Map2




tweets = tweets[tweets['lat'] >= 19.50139]
tweets = tweets[tweets['lat'] <= 64.85694]
tweets = tweets[tweets['lng'] >= -161.75583]
tweets = tweets[tweets['lng'] <= -68.01197]

tweets['date'] = pd.to_datetime(tweets['date']).dt.date

tweet_volume = pd.DataFrame(tweets['date'].value_counts().reset_index())
tweet_volume.columns = ['date', 'count']
tweet_volume = tweet_volume.sort_values('date', ascending=True)
fig = plt.figure(figsize =(7, 5))
plt.plot(tweet_volume['date'], tweet_volume['count'])
plt.xticks(rotation='vertical')
plt.xlabel("Dates")
plt.title("Has the volume of tweets been consistent over time?")
plt.ylabel("Tweet Volume")



source_counts = pd.DataFrame(tweets['source'].value_counts().reset_index())
source_counts.columns = ['source', 'count']

source_counts = source_counts.sort_values('count', ascending=False)
source_counts = source_counts.iloc[0:15]
fig = plt.figure(figsize =(10, 7))

sns.barplot(x = 'source',
            y = 'count',
            data = source_counts,
            palette = "Blues")

plt.xticks(rotation='vertical')
plt.xlabel("Source of Tweet")
plt.ylabel("Tweet Volume")
plt.title("What devices are people using?")
plt.show()



tags_counts = pd.DataFrame(tweets['hashtags'].value_counts().reset_index())
tags_counts.columns = ['hashtags', 'count']

tags_counts = tags_counts.sort_values('count', ascending=False)
tags_counts = tags_counts.iloc[0:15]
fig = plt.figure(figsize =(10, 5))

sns.barplot(x = 'hashtags',
            y = 'count',
            data = tags_counts,
            palette = "twilight")
plt.title("Commonly Used Hashtags")
plt.xlabel("Hashtags")
plt.ylabel("Tweet Volume")
plt.xticks(rotation='vertical')
plt.show()




tweets['clean_text'] = tweets['clean_text'].apply(lambda x: re.sub(r"https\S+", "", str(x)))
tweets['clean_text'] = tweets['clean_text'].apply(lambda x: x.lower())
tweets['clean_text'] = tweets['clean_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

stop_words = set(stopwords.words('english'))

tweets['clean_text'] = tweets['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

fig, img = plt.subplots(1,1,figsize=[7, 7])
wrdcld = WordCloud(width=500,height=500).generate(" ".join(tweets['clean_text']))

img.imshow(wrdcld)
img.set_title('Common Words')



word_counter = Counter([word for line in tweets['clean_text'] for word in line.split()]).most_common(70)

words_df = pd.DataFrame(word_counter[1:30])
words_df.columns = ['word', 'freq']

f, ax = plt.subplots(figsize=(8, 8))

sns.set_color_codes("pastel")
sns.barplot(x="freq", y="word", data=words_df,
            label="Total", color="b")
ax.set(ylabel="Words", xlabel="Frequency")
ax.set_title('Word Frequency')


nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
tweets['score'] = tweets['clean_text'].apply(lambda x: sid.polarity_scores(x))
tweets_new = pd.concat([tweets.drop(['score'], axis=1), tweets['score'].apply(pd.Series)], axis=1)
tweets_new['sentiment'] = tweets_new['compound'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))

tweets_pos = tweets_new[tweets_new['sentiment'] == 'positive']
tweets_neu = tweets_new[tweets_new['sentiment'] == 'neutral']
tweets_neg = tweets_new[tweets_new['sentiment'] == 'negative']

time_series_pos = pd.DataFrame(tweets_pos['date'].value_counts().reset_index())
time_series_pos.columns = ['date', 'count']
time_series_pos = time_series_pos.sort_values('date', ascending=True)

time_series_neu = pd.DataFrame(tweets_neu['date'].value_counts().reset_index())
time_series_neu.columns = ['date', 'count']
time_series_neu = time_series_neu.sort_values('date', ascending=True)

time_series_neg = pd.DataFrame(tweets_neg['date'].value_counts().reset_index())
time_series_neg.columns = ['date', 'count']
time_series_neg = time_series_neg.sort_values('date', ascending=True)

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
ax.plot(time_series_pos['date'], time_series_pos['count'], label='Positive')
ax.plot(time_series_neu['date'], time_series_neu['count'], label='Neutral')
ax.plot(time_series_neg['date'], time_series_neg['count'], label='Negative')

ax.set(title='Tweet Sentiments over Time', xlabel = 'Date', ylabel = 'Count')
ax.legend(loc='best')
fig.tight_layout()
plt.xlabel("Dates")
plt.ylabel("Tweet Volume")
plt.show()