# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:06:20 2022

@author: Akarsh Reddy
"""
from VM import VM,getResourcePool
from read_dataset import get0TimeStepJobs
resourcePool = getResourcePool()

# =============================================================================
# VMList contains list of all VM Objects
# =============================================================================
VMList = []
for i in resourcePool:
    VMList.append(VM(i.id,i.VMtype,i.vComp,i.vIO))
VMList[5].assignJob("lol")
for i in VMList:
    print(i.jobListEmpty())