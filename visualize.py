#Visualizer

import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

df = pd.read_csv('responsetimes.csv');

dictionary = {}
for i in range(200):
    dictionary[i]=[]

for i in range(0,200,1):
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
        
plt.plot(timeStepList,averageJobResponseTimes, color='g')
  
plt.xlabel("Timestep")
plt.ylabel("Avg job response time")
plt.title("Average Job Response Times - Round Robin")
  
plt.legend()
  
plt.show()       
print(numberOfRequests)
print(len(timeStepList))

plt.show()