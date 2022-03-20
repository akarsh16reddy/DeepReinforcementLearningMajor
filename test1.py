from VM import VM,getResourcePool
from read_dataset import get0TimeStepJobs
from Job import Job
from statistics import mean
import matplotlib.pyplot as plt
import random

resourcePool = getResourcePool()

print("id","type","vComp","vIO")
for i in resourcePool:
    print(i.id,i.VMtype,i.vComp,i.vIO,sep="   ")

VMList = []
for i in resourcePool:
    VMList.append(VM(i.id,i.VMtype,i.vComp,i.vIO))
    
for VMi in VMList:
    print(VMi.id,VMi.VMtype,VMi.vComp,VMi.vIO,sep=" ");
    
print("\n\n")
print("Printing 0th timestamp Jobs:")

pd0 = get0TimeStepJobs();

'''
for row in pd0.itertuples(index=True, name='Pandas'):
      print(row)
'''

print("\nAssigning jobs")
k=0
for i in range(0,203,1):
    jobList = []
    for row in pd0.loc[pd0['timestamp']==i].itertuples(index=True,name='Pandas'):
            jobList.append(Job(row.id,row.timestamp,row.type,row.rounded_load))
    
    for job in jobList:
        waitingTimes = []
        for vm in VMList:
            waitingTimes.append(vm.getRemainingExecutionTime())
        
        j=0
        temp = max(waitingTimes)
        if job.jType == 1:
           for m in range(0,5):
               if temp > waitingTimes[m]:
                   j = m
                   temp = waitingTimes[m]
        else:
            for m in range(5,10):
                if temp > waitingTimes[m]:
                    j = m
                    temp = waitingTimes[m]
        VMList[j].assignJob(job)
    for vm in VMList:
        vm.processJobs()

for i in VMList:
    i.printCurrentJobList()

for vm in VMList:
    if vm.jobListEmpty():
        print("VM ",vm.getId(),": done")
    else:
        print(print("VM ",vm.getId(),": not completed"))

'''
isDone = 0
while isDone==0:
    for vm in VMList:
        
'''   
'''        
#print("\nRemaining jobs")
#for i in VMList:
#    print("VM",i.getId(),i.getJobList(),sep=" ")

#print("\nRemaining jobs")
#for i in VMList:
#    if i.jobListEmpty()==False:
#        print("VM",i.getId(),i.getJobList(),sep=" ")

#dictionary = {}
#for i in range(200):
#    dictionary[i]=[]
    
#print("\n\nJob's and their response Times")
#for i in VMList:
#    print()
#    completedJobList = i.getCompletedJobQueue()
#    print("VM:",i.id,":",sep="",end="")
#    for job in completedJobList:
#       print("(",sep="",end="")
#        print(job.id,",",job.jobResponseTime,sep="",end="")
#        print(")",sep="",end="")
#        dictionary[job.arrivalT].append(job.jobResponseTime)

#print(dictionary)
averageJobResponseTimes = []
numberOfRequests = []
for key,value in dictionary.items():
    numberOfRequests.insert(key,len(value))
    if len(value)!=0:
        averageJobResponseTimes.insert(key,mean(value))
    else:
        averageJobResponseTimes.insert(key,0)

#timeStepList = [i for i in range(0,200)]
#plt.plot(timeStepList,averageJobResponseTimes)
#plt.show()
'''
'''
print("\nAssigning jobs in round-robin fashion")
i=0;
for row in jobList:
    VMList[i].assignJob(row)
    i=i+1
    i=i%10

print("\n\nPrinting job queues of all VMs")
for i in VMList:
    print("VM",i.getId(),i.getJobList(),sep=" ")
    
print("\nProcess jobs:")
for i in VMList:
    i.processJobs()
for i in VMList:
    if not i.jobListEmpty():
        print("VM",i.getId(),i.getJobList(),sep=" ")'''