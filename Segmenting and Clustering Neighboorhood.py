#!/usr/bin/env python
# coding: utf-8

# # Question 1 <a id="0"></a>

# 1.Getting libraries

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

from bs4 import BeautifulSoup
import requests

print('Libraries imported.')


# 2.Building code to scrape the following Wikipedia page: https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M

# In[2]:


toronto_data = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M' 
toronto_dataextracted = requests.get(toronto_data).text
soup = BeautifulSoup(toronto_dataextracted, 'lxml')
toronto_table = soup.find('table',{'class':'wikitable sortable'})
toronto_table


# In[3]:


headers="Postcode,Borough,Neighbourhood"


# In[4]:


#toronto_table = pd.DataFrame(data, columns=['Postalcode', 'Borough', 'Neighbourhood'])
#toronto_table = toronto_table[~toronto_table['Postalcode'].isnull()]
#toronto_table.drop(toronto_table[toronto_table['Borough']=="Not assigned"].index,axis=0, inplace=True)
#toronto_table.head()
data=""
for tr in toronto_table.find_all('tr'):
    row1=""
    for tds in tr.find_all('td'):
        row1=row1+","+tds.text
    data=data+row1[1:]
print(data)


# In[5]:


new=open("toronto.csv","wb")
new.write(bytes(data,encoding="ascii",errors="ignore"))


# In[6]:


toronto_new = pd.read_csv('toronto.csv',header=None)
toronto_new.columns=["Postalcode","Borough","Neighbourhood"]
toronto_new.head(15)


# In[7]:


index_df = toronto_new[ toronto_new['Borough'] =='Not assigned'].index
toronto_new.drop(index_df , inplace=True)
toronto_new.loc[toronto_new['Neighbourhood'] =='Not assigned' , 'Neighbourhood'] = toronto_new['Borough']
result = toronto_new.groupby(['Postalcode','Borough'], sort=False).agg( ', '.join)
toronto_new2=result.reset_index()
toronto_new2.head(15)


# In[8]:


print("the dataframe size is:", toronto_new2.size)
print("the dataframe info is:", toronto_new2.info)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Question 2 <a id="0"></a>

# 1.-getting coordinates

# In[16]:


get_ipython().system("wget -q -O 'Toronto_coord.csv'  http://cocl.us/Geospatial_data")
df_coord = pd.read_csv('Toronto_coord.csv')
df_coord.head()


# In[17]:


df_coord.columns=['Postalcode','Latitude','Longitude']
df_coord.head()


# we are going to merge both dataframes (toronto_new2 and df_coord)

# In[19]:


df_merged = pd.merge(toronto_new2,
                 df_coord[['Postalcode','Latitude', 'Longitude']],
                 on='Postalcode')
df_merged


# In[20]:


print("the dataframe size is:", df_merged.size)
print("the dataframe info is:", df_merged.info)


# # Question 3 <a id="0"></a>

# 1.-extracting libraries needed

# In[21]:


# import k-means from clustering stage
from sklearn.cluster import KMeans
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library


# In[22]:


from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[23]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="Toronto")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# 2.-Creating a map of Toronto with neighborhoods superimposed on top.

# In[27]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_merged['Latitude'], df_merged['Longitude'], df_merged['Borough'], df_merged['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# 3.-Getting Foursquare credentials

# In[29]:


CLIENT_ID = 'HYA5PUMBLRHBCJYOIANTFG1QYS3W4DZIYYYFVSVTJNNJQ0XY' # your Foursquare ID
CLIENT_SECRET = '0UYOK4AJCKHACBATY2ABEH2LZ4J1U4LWV5OO0MOCX3R5I14N' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# 4.- Now, let's get the top 100 venues that are in Toronto within a radius of 500 meters.

# In[34]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    latitude, 
    longitude, 
    radius, 
    LIMIT)
url # display URL


# 5.- Getting results

# In[35]:


results = requests.get(url).json()
results


# In[36]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[37]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# 6.- Getting the number of venues in Toronto

# In[38]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# 7.-Exploring Toronto Neighborhood, getting number of venues nerby

# In[39]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Latitude', 
                  'Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[41]:


Toronto_venues = getNearbyVenues(names=df_merged['Neighbourhood'],
                                   latitudes=df_merged['Latitude'],
                                   longitudes=df_merged['Longitude']
                                  )
print(Toronto_venues.shape)
Toronto_venues.head()


# In[44]:


Toronto_venues.groupby('Neighborhood').count()


# In[45]:


print('There are {} uniques categories.'.format(len(Toronto_venues['Venue Category'].unique())))


# 8.-Analyzing each neighborhood

# In[82]:


# one hot encoding
toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# 9.-grouping rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[83]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# 10-printing each neighborhood along with the top 5 most common venues

# In[85]:


num_top_venues = 5

for hood in toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[73]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[86]:


import numpy as np
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()


# 11.-Clustering Neighborhoods

# In[87]:


# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_


# In[88]:


neighbourhoods_venues_sorted.insert(0, 'Cluster_Labels', kmeans.labels_)

toronto_merged = df_merged

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head() # check the last columns!


# In[77]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Cluster 1

# In[ ]:


df_merged.loc[df_merged['Cluster Labels'] == 0, df_merged.columns[[1] + list(range(5, df_merged.shape[1]))]]

