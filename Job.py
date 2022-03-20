# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:33:49 2022

@author: Akarsh Reddy
"""
from csv import writer
    
class Job:
    def __init__(self, id, arrivalT, jType, jSize):
        self.id= id
        self.arrivalT = arrivalT
        self.jType = jType
        self.jSize = jSize
        self.jobResponseTime = 0
        self.executionTime = 0
        self.waitTime = 0
        self.originalSize = jSize
    
    def getId(self):
        return self.id
    
    def getArrivalT(self):
        return self.arrivalT 
    
    def getjType(self):
        return self.jType
    
    def getjSize(self):
        return self.jSize
    
    def getOriginalSize(self):
        return self.originalSize
    
    def updateWaitTime(self,time):
        self.waitTime = time
        
    def updateExecutionTime(self,time):
        self.executionTime = time
        
    def updateResponseTime(self):
        self.jobResponseTime = self.waitTime + self.executionTime

        with open('responsetimes_earliest.csv', 'a',newline='',) as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([self.id,self.arrivalT,self.jobResponseTime])
            f_object.close()
            
    def getDetails(self):
        return str(self.id)+" "+str(self.arrivalT)+" "+str(self.jType)+" "+str(self.jSize)

        