
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd


# In[3]:

cab_data = pd.read_csv('yellow_tripdata_2016-03.csv')


# In[88]:

trips = cab_data.loc[:,('tpep_pickup_datetime','passenger_count','pickup_longitude','pickup_latitude')]
print(trips[0:3])


# In[89]:

trips["pickupdate"], trips["pickuptime"] = zip(*trips["tpep_pickup_datetime"].str.split().tolist())
del trips["tpep_pickup_datetime"]  


# In[90]:

mar_hol = ['2016-03-05','2016-03-06','2016-03-12','2016-03-13','2016-03-19','2016-03-20','2016-03-26','2016-03-27']
trips = trips.loc[trips['pickupdate'].isin(mar_hol)]
trips['timegroup'] = trips['pickuptime']
print(trips[0:5])


# In[91]:

pt = trips.ix[(trips["pickup_longitude"] > -73.993977)]


# In[92]:

pent = pt.ix[(pt["pickup_longitude"] < -73.991983)]


# In[93]:

pennt = pent.ix[(pent["pickup_latitude"] > 40.750277)]


# In[94]:

df = pennt.ix[(pennt["pickup_latitude"] < 40.750341)]


# In[95]:

print(df[0:5])


# In[96]:

pd.options.mode.chained_assignment = None
df['pickuptime'] = df['pickuptime'].map(lambda x: str(x)[:-3])


# In[97]:

print(df[0:20])


# In[98]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[99]:

grouped = df.groupby('pickuptime')
print(grouped)


# In[115]:

grouped.sum()


# In[238]:

result = df[['pickuptime','passenger_count','timegroup']]
result


# In[284]:

train_times = pd.read_csv('lirrtime.csv')
train_times = train_times.dropna(thresh=1)


# In[285]:

from datetime import datetime
import matplotlib.dates as mdate 
pd.to_datetime(result['timegroup'])
import matplotlib.dates as mdates
print(result[0:5])


# In[286]:

plt.figure(1)
x = result['timegroup']
y = result['passenger_count']
plt.plot_date(x,y,'bo')

plt.xlabel('Time')
plt.gcf().autofmt_xdate()

plt.ylabel('Sum of Passengers')

traintimes = train_times['Time']
for arrival in traintimes:
    pd.to_datetime(arrival)
    plt.axvline(arrival, color= 'k', linestyle= '-')
    
fig1 = plt.savefig('myPlot.png')
fig1


# In[397]:

cab = result['pickuptime']
print(cab)
print(traintimes)


# In[398]:

def get_sec(time_str):
    h, m = map(int,time_str.split(':'))
    return int(h) * 3600 + int(m) * 60

traintimes = traintimes.astype(str)


# In[399]:

lirr = []
for a in traintimes:
    lirr.append(get_sec(a))

tt = []
for li in lirr:
    tt.append(range(li,li+600))

all_train = []
for t in tt:
    for n in t:
        all_train.append(n)


# In[400]:

cab = cab.apply(get_sec,1)
cab


# In[401]:

def traintransfer(PUtime):
    if cab in all_train:
        result['LIRR'] = 1
    else: result['LIRR'] = 0


# In[410]:

result['LIRR'] = cab.isin(all_train)

result


# In[437]:

group2 = result.groupby('LIRR')
result2 = group2.sum()
print(group2.sum())
print(group2.size())

def get_avg(x):
    if result2 == True:
        result2['avg'] = 62/38
    else:
        result2['avg'] = 112/83


# In[449]:

avg = pd.DataFrame({'Riders':('All','Tunnels'), 'Passenger Count': (172, 62), 'Count':(121,38)})
avg


# In[450]:

avg['Avg Passenger'] = avg['Passenger Count'] / avg['Count']
avg


# In[456]:

y = avg['Avg Passenger']
n = len(y)
x = range(n)
bar_width = 0.35
opacity = 0.4
index = np.arange(n)

plt.bar(x,y,width,color='blue')
plt.xlabel('Group')
plt.ylabel('Avg. Passenger')
plt.title('Average Number of Passengers')
plt.xticks(index + bar_width, ('All', 'Tunnels'))


# In[ ]:



