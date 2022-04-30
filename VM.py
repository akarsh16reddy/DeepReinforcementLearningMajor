# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:43:28 2022

@author: Akarsh Reddy
"""
from Job import Job

class VM:
    def __init__(self, id, VMtype, vComp, vIO):
        self.id = id
        self.VMtype = VMtype
        self.vComp = vComp
        self.vIO = vIO
        self.jobQueue = []
        self.timeStep = 0
        self.completedJobQueue = []
        self.remainingExecutionTime = 0
        self.expectedWaitTime = 0
        
    def getCompletedJobQueue(self):
        return self.completedJobQueue 
        
    def assignJob(self,Jobx):
        self.jobQueue.append(Jobx)
    
    def getId(self):
        return self.id
    
    def getJobList(self):
        return [(j.getId(),j.getjSize()) for j in self.jobQueue]
    
    def getRemainingExecutionTime(self):
        if self.jobListEmpty():
            return 0
        
        remainingExecutionTime = 0.0
        for job in self.jobQueue:
            if job.jType==0:
                remainingExecutionTime += job.jSize/self.vIO
            else:
                if job.jType==1:
                    remainingExecutionTime += job.jSize/self.vComp
        return remainingExecutionTime
        
    def jobListEmpty(self):
        '''
        setOfCompletedJobs = set(self.completedJobQueue)
        tempList = [x for x in self.jobQueue if x not in setOfCompletedJobs]
        if len(tempList)==0:
            return True
        else:
            return False
        '''
        if len(self.jobQueue)==0:
            return True
        else:
            return False
        
    def printCurrentJobList(self):
        '''
        if self.timeStep%5==0:
            tempList = [job for job in self.jobQueue if job.jSize>0]
            print("VM ",self.id," :",end="\n")
            for job in tempList:
                print("(",end="",sep="")
                print(job.id,",",job.jSize,end="",sep="")
                print(")",end="",sep="")
            print()
        '''
        print("\n\nVM ",self.id,end="\n")
        for job in self.jobQueue:
            print("[",end="",sep="")
            print(job.id,",",job.jSize,end="",sep="")
            print("]",end="",sep="")
            
    def processJobs(self):
# =============================================================================
#         for j in self.jobQueue:
#             if j.jType==1:
#                 j.updateExecutionTime(j.jSize/self.vComp)
#             else:
#                 j.updateExecutionTime(j.jSize/self.vIO)
# =============================================================================
        '''MIPSPercentLeft = 1
        for i,j in enumerate(self.jobQueue):
            if j.jType == 1:
                effectivevComp = MIPSPercentLeft*self.vComp
                if j.jSize > effectivevComp:
                    self.jobQueue[i].jSize -= (effectivevComp)
                    break
                else if j.jSize < MIPSPercentLeft*vComp:
                    MIPSPercentLeft = 1-(j.jSize/(MIPSPercentLeft*vComp))
                    del self.jobQueue[i]
            else if j.jType == 0:
                self.jobQueue.jSize -= self.vIO
            '''
        if self.jobListEmpty():
            return
        effectivevComp = self.vComp
        effectivevIO = self.vIO
        timeConsumed = 0 
        while effectivevComp!=0 and effectivevIO!=0:
            if self.jobListEmpty():
               break
            j = self.jobQueue[0]
            if j.jType==1 and j.jSize>0:
                if j.jSize > effectivevComp:
                    self.jobQueue[0].jSize -= effectivevComp
                    effectivevComp=0
                else:
                    timeLeft = 1-(j.jSize/effectivevComp)
                    timeConsumed = 1 - timeLeft
                    effectivevComp *= timeLeft
                    effectivevIO *= timeLeft
                    self.jobQueue.pop(0)
            if j.jType==0 and j.jSize>0:
                if j.jSize > effectivevIO:
                    self.jobQueue[0].jSize -= effectivevIO
                    effectivevIO=0
                else:
                    timeLeft = 1-(j.jSize/effectivevIO)
                    timeConsumed = 1 - timeLeft
                    effectivevComp *= timeLeft
                    effectivevIO *= timeLeft
                    self.jobQueue.pop(0)
                        
#        tempList = [job for job in self.jobQueue if job.jSize>0]
#        self.jobQueue = tempList
        self.timeStep += 1
    
CI_VM = [(1,970,498),
         (2,930,467),
         (3,1098,536),
         (4,1192,466),
         (5,1175,543)]

IOI_VM = [(6,573,894),
          (7,444,1012),
          (8,427,1038),
          (9,505,1065),
          (10,563,1096)]

# =============================================================================
# CI_VM_test = [(i,100,100) for i in range(0,5)]
# IOI_VM_test = [(i,100,100) for i in range(5,10)]
# 
# 
# =============================================================================
CI_VM_Objs = [VM(i,1,j,k) for i,j,k in CI_VM]
IOI_VM_Objs=[VM(i,0,j,k) for i,j,k in IOI_VM]

# =============================================================================
# CI_VM_test_Objs = [VM(i,1,j,k) for i,j,k in CI_VM_test]
# IOI_VM_test_Objs=[VM(i,0,j,k) for i,j,k in IOI_VM_test]
# #for i in IOI_VM_Objs:
# #   print(i.id,i.VMtype,i.vComp,i.vIO,sep=" ");
# =============================================================================
    
def getResourcePool():
    return CI_VM_Objs+IOI_VM_Objs
    