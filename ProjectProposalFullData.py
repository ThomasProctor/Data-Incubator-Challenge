
# coding: utf-8

# In[33]:

from __future__ import division
import re
import urllib2
import pandas as pd
import StringIO
import zipfile
import requests
import numpy as np
import pickle
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt


# In[2]:

path = 'https://s3.amazonaws.com/tripdata/'
response = urllib2.urlopen(path).read()


# In[3]:

regex = re.compile('2015\d{1,2}-citibike-tripdata.zip')


# In[4]:

files = regex.findall(response)


# In[5]:

def zip_to_csv(zip_name):
    base_name = zip_name[:zip_name.find('.zip')]
    return base_name + '.csv'


# In[6]:

csv_names = [zip_to_csv(name) for name in files]


# In[7]:

def zipped_online_csv_read(url, csv_filename, **kwargs):
    s = requests.get(url).content
    z = zipfile.ZipFile(StringIO.StringIO(s))
    return pd.read_csv(z.open(csv_filename), **kwargs)


# In[8]:

columns = ("start station id, end station id, "
           "start station latitude, start station longitude, "
           "end station latitude, tripduration, "
           "end station longitude, usertype").split(', ')


# In[9]:

kwargs = {'usecols':columns}


# In[10]:

df = pd.DataFrame(columns=columns)
counter = 0
for i in zip(files, csv_names):
    file_path = path + i[0]
    df = df.append(zipped_online_csv_read(file_path, i[1],
                                          **kwargs))
    print(i[1])
    print(df.shape)
    counter += 1
    


# ## Getting google directions data

# In[11]:

west_side_hwy_stations = [72, 458, 459, 478, 508, 525, 530,
                          514]
first_ave_stations = [2012, 2003, 545, 536, 527, 516, 504, 455,
                      454, 438, 428, 326, 174]


# In[12]:

with open('west_side.pkl', 'r') as f:
    west_side = pickle.load(f)


# In[13]:

west_side_array = np.array([[j['distance']['value']
                              for j in i['elements']]
                             for i in west_side['rows']])

west_side_time = np.array([[j['duration']['value']
                              for j in i['elements']]
                             for i in west_side['rows']])


# In[14]:

west_side_df = pd.DataFrame(west_side_array,
                            index=west_side_hwy_stations,
                            columns=west_side_hwy_stations)

west_side_time = pd.DataFrame(west_side_time,
                            index=west_side_hwy_stations,
                            columns=west_side_hwy_stations)


# In[15]:

with open('cross_town.pkl', 'r') as f:
    cross_town = pickle.load(f)


# In[16]:

cross_town_array = np.array([[j['distance']['value']
                              for j in i['elements']]
                             for i in cross_town['rows']])

cross_town_time = np.array([[j['duration']['value']
                              for j in i['elements']]
                             for i in cross_town['rows']])


# In[17]:

cross_town_df = pd.DataFrame(cross_town_array,
                            index=west_side_hwy_stations,
                            columns=first_ave_stations[:8])

cross_town_time = pd.DataFrame(cross_town_time,
                            index=west_side_hwy_stations,
                            columns=first_ave_stations[:8])


# ## Merging in google maps data

# In[18]:

use_trip = ((df['start station id']
             .isin(west_side_hwy_stations))
            & (df['end station id']
              .isin(west_side_hwy_stations
                    + first_ave_stations[:8]))
            & (df['start station id']
               != df['end station id']))


# In[19]:

df = df[use_trip]


# In[20]:

west_trips = df[(df['start station id']
                 .isin(west_side_hwy_stations))
                & (df['end station id']
                  .isin(west_side_hwy_stations))]
def distance_lookup(ser, table):
    return table.loc[ser['start station id'],
                     ser['end station id']]

west_trips['googledistance'] = (west_trips
                                 .apply(lambda x: distance_lookup(x, west_side_df),
                                        axis=1))


# In[21]:

cross_trips = df[(df['start station id']
                  .isin(west_side_hwy_stations))
                 & (df['end station id']
                    .isin(first_ave_stations[:8]))]
cross_trips['googledistance'] = (cross_trips
                                 .apply(lambda x: distance_lookup(x, cross_town_df),
                                        axis=1))


# In[22]:

df = west_trips.append(cross_trips)


# In[23]:

df['crosstown'] = df['end station id'].isin(first_ave_stations[:8])


# In[24]:

reasonable_distance = df['googledistance'] > 1000


# In[25]:

df = df[reasonable_distance]


# In[31]:

group = df.groupby(['start station id', 'end station id'])
mean_df = group.mean()


# In[ ]:

def ci(s):
    vals = s.values
    ci = sms.DescrStatsW(vals).tconfint_mean()
    return pd.Series(ci)
confidence_intervals = group['tripduration'].apply(ci)
confidence_intervals = confidence_intervals.unstack()
names = {0:'ci95% low', 1:'ci95% high'}
confidence_intervals.rename(columns=names, inplace=True)
mean_df = mean_df.join(confidence_intervals)


# In[60]:

mean_df.dropna(axis='index', inplace=True)


# In[76]:

mean_model = smf.ols(formula='tripduration ~ googledistance * crosstown + googledistance + crosstown + 1', data=mean_df)
mean_results = mean_model.fit()
print('\nMean model results:')
print(mean_results.summary())


# In[77]:

smean_model = smf.ols(formula='tripduration ~  googledistance + crosstown + 1', data=mean_df)
smean_results = smean_model.fit()
print('\nSimple Mean model results:')
print(smean_results.summary())


# In[30]:

mult_model = smf.ols(formula='tripduration ~ googledistance * crosstown + googledistance + crosstown + 1', data=df)
mult_results = mult_model.fit()
print('Multiplicative model, results:')
print(mult_results.summary())


# In[31]:

simple_model = smf.ols(formula='tripduration ~ googledistance + crosstown + 1', data=df)
simple_results = simple_model.fit()
print('Simple model, results:')
print(simple_results.summary())


# ## Plot results

# In[74]:

XX = np.arange(mean_df['googledistance'].min() - 400,
               mean_df['googledistance'].max() + 400)

ctXX = pd.DataFrame({'googledistance':XX,
                     'crosstown':np.ones(XX.shape)})

wsXX = pd.DataFrame({'googledistance':XX,
                     'crosstown':np.zeros(XX.shape)})

scatter_kwargs = {"zorder":0, 'alpha':0.3, 'fmt':'o'}
line_kwargs = {"zorder":100, 'alpha':0.8, 'lw':2}

a=6
gr=(1+np.sqrt(5))/2
plt.figure(figsize=[a*gr,a])
plt.autoscale(tight=True)
plt.xlabel('Google maps distance, meters')
plt.ylabel('Recorded trip duration, seconds')
plt.title("Trip distance vs time")
colorlist=plt.cm.gist_rainbow(np.linspace(0,2.8/4,4)).tolist()
colorlist.reverse()
cross_trips = mean_df[mean_df['crosstown'] == True]
plt.errorbar(cross_trips['googledistance'],
             cross_trips['tripduration'],
             label='cross town trips\n95% confidence\ninterval',
             color=colorlist[0],
             yerr=[cross_trips['ci95% low'], cross_trips['ci95% high']],
             **scatter_kwargs)
wests_trips = mean_df[mean_df['crosstown'] == False]
plt.errorbar(wests_trips['googledistance'],
             wests_trips['tripduration'],
             label='Hudson River\nGreenway trips\n95% confidence\ninterval',
             yerr=[wests_trips['ci95% low'], wests_trips['ci95% high']],
             color=colorlist[-1], **scatter_kwargs)
plt.plot(XX, mean_results.predict(ctXX), color=colorlist[0],
         label='Fit, cross town data', **line_kwargs)
plt.plot(XX, mean_results.predict(wsXX), color=colorlist[-1],
         label='Fit, Hudson River\nGreenway data',
         **line_kwargs)
plt.ylim(-100, 2000)
plt.legend(loc='lower right')
plt.savefig('trip_distance_vs_time_actual.pdf')
#plt.show()


# In[78]:

XX = np.arange(mean_df['googledistance'].min() - 400,
               mean_df['googledistance'].max() + 400)

ctXX = pd.DataFrame({'googledistance':XX,
                     'crosstown':np.ones(XX.shape)})

wsXX = pd.DataFrame({'googledistance':XX,
                     'crosstown':np.zeros(XX.shape)})

scatter_kwargs = {"zorder":0, 'alpha':0.3, 'fmt':'o'}
line_kwargs = {"zorder":100, 'alpha':0.8, 'lw':2}

a=6
gr=(1+np.sqrt(5))/2
plt.figure(figsize=[a*gr,a])
plt.autoscale(tight=True)
plt.xlabel('Google maps distance, meters')
plt.ylabel('Recorded trip duration, seconds')
plt.title("Trip distance vs time")
colorlist=plt.cm.gist_rainbow(np.linspace(0,2.8/4,4)).tolist()
colorlist.reverse()
cross_trips = mean_df[mean_df['crosstown'] == True]
plt.errorbar(cross_trips['googledistance'],
             cross_trips['tripduration'],
             label='cross town trips\n95% confidence\ninterval',
             color=colorlist[0],
             yerr=[cross_trips['ci95% low'], cross_trips['ci95% high']],
             **scatter_kwargs)
wests_trips = mean_df[mean_df['crosstown'] == False]
plt.errorbar(wests_trips['googledistance'],
             wests_trips['tripduration'],
             label='Hudson River\nGreenway trips\n95% confidence\ninterval',
             yerr=[wests_trips['ci95% low'], wests_trips['ci95% high']],
             color=colorlist[-1], **scatter_kwargs)
plt.plot(XX, smean_results.predict(ctXX), color=colorlist[0],
         label='Fit, cross town data', **line_kwargs)
plt.plot(XX, smean_results.predict(wsXX), color=colorlist[-1],
         label='Fit, Hudson River\nGreenway data',
         **line_kwargs)
plt.ylim(-100, 2000)
plt.legend(loc='lower right')
plt.savefig('simple_trip_distance_vs_time_actual.pdf')
#plt.show()


# In[ ]:



