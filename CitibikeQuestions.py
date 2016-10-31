
# coding: utf-8


from __future__ import division
import re
import urllib2
import pandas as pd
import StringIO
import zipfile
import requests
import numpy as np




path = 'https://s3.amazonaws.com/tripdata/'
response = urllib2.urlopen(path).read()




regex = re.compile('2015\d{1,2}-citibike-tripdata.zip')




files = regex.findall(response)




def zip_to_csv(zip_name):
    base_name = zip_name[:zip_name.find('.zip')]
    return base_name + '.csv'




csv_names = [zip_to_csv(name) for name in files]




def zipped_online_csv_read(url, csv_filename, **kwargs):
    s = requests.get(url).content
    z = zipfile.ZipFile(StringIO.StringIO(s))
    return pd.read_csv(z.open(csv_filename), **kwargs)




columns = ("tripduration, start station id, end station id, "
           "start station latitude, start station longitude, "
           "end station latitude, end station longitude, "
           "bikeid, starttime, usertype").split(', ')
dates = ['starttime']




kwargs = {'usecols':columns, 'parse_dates':dates,
          'infer_datetime_format':True}




df = pd.DataFrame(columns=columns)
counter = 0
for i in zip(files, csv_names):
    file_path = path + i[0]
    df = df.append(zipped_online_csv_read(file_path, i[1],
                                          **kwargs))
    print(i[1])
    print(df.shape)
    counter += 1
    






question = "What is the median trip duration, in seconds?\n{:5.10f}"
print(question.format(int(df['tripduration'].median())))






question = ("What fraction of rides start and end "
            "at the same station?\n{:5.10f}")
same = df[df['start station id']
          == df['end station id']].shape[0]
print(question.format(same / df.shape[0]))






question = ("We say a bike has visited a station if it has "
            "a ride that either started or ended at that "
            "station. Some bikes have visited many stations; "
            "others just a few. What is the standard deviation "
            "of the number of stations visited by a bike?\n{:5.10f}")
def stations_visited(grp_df):
    started = set(grp_df['start station id'].dropna().unique())
    ended = set(grp_df['end station id'].dropna().unique())
    return len(started | ended)
group = df.groupby('bikeid')
bike_stations_visited = group.apply(stations_visited)
print(question.format(bike_stations_visited.std()))






question = ("What is the average length, in kilometers, of a "
            "trip? Assume trips follow great circle arcs from "
            "the start station to the end station. Ignore "
            "trips that start and end at the same station, as "
            "well as those with obviously wrong data.\n{:5.10f}")

def great_circle_dist(lat1, lon1, lat2, lon2):
    radius = 6371 #earth's radius in km
    degree_radian_ratio = np.pi / 180
    phi1 = degree_radian_ratio * lat1
    phi2 = degree_radian_ratio * lat2
    dphi = phi2 - phi1
    dtheta = degree_radian_ratio * (lon2 - lon1)
    a = (np.sin(dphi / 2)**2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dtheta / 2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

def mean_dist(df):
    good_df = df[df['start station id']
                 != df['end station id']]
    dists = great_circle_dist(good_df['start station latitude'],
                              good_df['start station longitude'],
                              good_df['end station latitude'],
                              good_df['end station longitude'])
    return dists.mean()
    

print(question.format(mean_dist(df)))
    
    






question = ("Calculate the average duration of trips for "
            "each month in the year. (Consider a trip to "
            "occur in the month in which it starts.) What "
            "is the difference, in seconds, between the "
            "longest and shortest average durations?\n{:5.10f}")

group = df.groupby(df['starttime'].dt.month)
means = group['tripduration'].mean()
diff = means.max() - means.min()
print(question.format(diff))






question = ('Let us define the hourly usage fraction of a '
            'station to be the fraction of all rides '
            'starting at that station that leave during '
            'a specific hour. A station has surprising usage '
            'patterns if it has an hourly usage fraction for '
            'an hour significantly different from the '
            'corresponding hourly usage fraction of the '
            'system as a whole. What is the largest ratio of '
            'station hourly usage fraction to system hourly '
            'usage fraction (hence corresponding to the most '
            '"surprising" station-hour pair)?\n{:5.10f}')

hour = df['starttime'].dt.hour

group = df.groupby([df['start station id'],
                    hour])

total = df.groupby('start station id')['bikeid'].count()

station_hourly = group['bikeid'].count() / total

total_hourly = df.groupby(hour).count()['bikeid']

def divide_total(df):
    return df.divide(total_hourly, fill_value=0.0)

top_group = station_hourly.groupby(level=['start station id'])
station_system = top_group.apply(divide_total)

print(question.format(station_system.max()))






question = ('There are two types of riders: "Customers" and '
            '"Subscribers." Customers buy a short-time pass '
            'which allows 30-minute rides. Subscribers buy '
            'yearly passes that allow 45-minute rides. What '
            'fraction of rides exceed their corresponding '
            'time limit?\n{:5.10f}')
subscribers_over = ((df['usertype'] == 'Subscriber')
                    & (df['tripduration'] > 45 * 60))
customers_over = ((df['usertype'] == 'Customer')
                    & (df['tripduration'] > 30 * 60))
exceeded_frac = (customers_over | subscribers_over).mean()
print(question.format(exceeded_frac))






question = ("Most of the time, a bike will begin a trip at "
            "the same station where its previous trip ended. "
            "Sometimes a bike will be moved by the program, "
            "either for maintenance or to rebalance the "
            "distribution of bikes. What is the average "
            "number of times a bike is moved during this "
            "period, as detected by seeing if it starts at a "
            "different station than where the previous ride "
            "ended?\n{:5.10f}")
group = df.groupby('bikeid')
def times_moved(df):
    end_stations = df['end station id'].values
    rolled_stations = np.roll(end_stations, 1)
    start_stations = df['start station id'].values
    return (rolled_stations[1:] != start_stations[1:]).sum()
mean_moved = group.apply(times_moved).mean()
print(question.format(mean_moved))
    






