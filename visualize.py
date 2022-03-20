#Visualizer

import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

df = pd.read_csv('responsetimes_random.csv');

dictionary = {}
for i in range(198):
    dictionary[i]=[]

for i in range(0,198,1):
    for row in df.loc[df['arrivalT']==i].itertuples(index=True,name='Pandas'):
        dictionary[row.arrivalT].append(row.responseTime)

averageJobResponseTimes = []
numberOfRequests = []
timeStepList = []
for key,value in dictionary.items():
    numberOfRequests.insert(key,len(value))
    timeStepList.append(key)
    if len(value)!=0:
        averageJobResponseTimes.insert(key,mean(value))
    else:
        averageJobResponseTimes.insert(key,0)

plt.plot(timeStepList,averageJobResponseTimes, color='y', label="random")
plt.xlabel("Timestep")
plt.ylabel("Avg job response time")
plt.title("Average Job Response Times")

df = pd.read_csv('responsetimes_roundrobin.csv');

dictionary = {}
for i in range(198):
    dictionary[i]=[]

for i in range(0,198,1):
    for row in df.loc[df['arrivalT']==i].itertuples(index=True,name='Pandas'):
        dictionary[row.arrivalT].append(row.responseTime)

averageJobResponseTimes_rr = []
numberOfRequests_rr = []
timeStepList_rr = []
for key,value in dictionary.items():
    numberOfRequests_rr.insert(key,len(value))
    timeStepList_rr.append(key)
    if len(value)!=0:
        averageJobResponseTimes_rr.insert(key,mean(value))
    else:
        averageJobResponseTimes_rr.insert(key,0)
plt.plot(timeStepList_rr,averageJobResponseTimes_rr, color='b',label="round robin")
  
plt.legend()
  
plt.show()       
print(numberOfRequests)
print(len(timeStepList))

plt.show()