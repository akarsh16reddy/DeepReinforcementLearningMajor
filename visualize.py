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

plt.plot(timeStepList,averageJobResponseTimes, color='deepskyblue', label="random")
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

plt.plot(timeStepList_rr,averageJobResponseTimes_rr, color='orange',label="round robin")
  
df = pd.read_csv('responsetimes_earliest.csv');

dictionary = {}
for i in range(198):
    dictionary[i]=[]

for i in range(0,198,1):
    for row in df.loc[df['arrivalT']==i].itertuples(index=True,name='Pandas'):
        dictionary[row.arrivalT].append(row.responseTime)

averageJobResponseTimes_earliest = []
numberOfRequests_earliest = []
timeStepList_earliest = []
for key,value in dictionary.items():
    numberOfRequests_earliest.insert(key,len(value))
    timeStepList_earliest.append(key)
    if len(value)!=0:
        averageJobResponseTimes_earliest.insert(key,mean(value))
    else:
        averageJobResponseTimes_earliest.insert(key,0)
print(max(averageJobResponseTimes_earliest))
plt.plot(timeStepList_earliest,averageJobResponseTimes_earliest, color='green',label="earliest")
  
df = pd.read_csv('responsetimes_best_fit.csv');

dictionary = {}
for i in range(198):
    dictionary[i]=[]

for i in range(0,198,1):
    for row in df.loc[df['arrivalT']==i].itertuples(index=True,name='Pandas'):
        dictionary[row.arrivalT].append(row.responseTime)

averageJobResponseTimes_best = []
numberOfRequests_best = []
timeStepList_best = []
for key,value in dictionary.items():
    numberOfRequests_best.insert(key,len(value))
    timeStepList_best.append(key)
    if len(value)!=0:
        averageJobResponseTimes_best.insert(key,mean(value))
    else:
        averageJobResponseTimes_best.insert(key,0)
plt.plot(timeStepList_best,averageJobResponseTimes_best, color='mediumorchid',label="best fit")
print(max(averageJobResponseTimes_best))


df = pd.read_csv('responsetimes_DRL.csv');

dictionary = {}
for i in range(198):
    dictionary[i]=[]

for i in range(0,198,1):
    for row in df.loc[df['arrivalT']==i].itertuples(index=True,name='Pandas'):
        dictionary[row.arrivalT].append(row.responseTime)
        
averageJobResponseTimes_drl = []
numberOfRequests_drl = []
timeStepList_drl = []
for key,value in dictionary.items():
    numberOfRequests_drl.insert(key,len(value))
    timeStepList_drl.append(key)
    if len(value)!=0:
        averageJobResponseTimes_drl.insert(key,mean(value))
    else:
        averageJobResponseTimes_drl.insert(key,0)
plt.plot(timeStepList_drl,averageJobResponseTimes_drl, color='red',label="DRL")

plt.legend()
plt.show()       
print(numberOfRequests)
print(len(timeStepList))

plt.show()