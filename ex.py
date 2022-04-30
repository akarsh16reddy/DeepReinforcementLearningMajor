# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:03:28 2022

@author: Akarsh Reddy
"""
# =============================================================================
# lst1= [1,2,3]
# lst2 =[1,2,3,4]
# 
# print(set(lst1) ^ set(lst2))
# 
# import numpy as np
# 
# print(np.random.random())
# 
# print(np.random.randint(9,11))
# 
# from collections import deque
# replay_memory = deque(maxlen=3)
# replay_memory.append(1)
# replay_memory.append(2)
# replay_memory.append(3)
# replay_memory.append(4)
# replay_memory.append(5)
# 
# print(replay_memory)
# 
# current_state=np.array([0]*12)
# np.put(current_state, [range(2,12)], [100, 1000])
# print(current_state)
# 
# from tqdm import tqdm
# for i in tqdm(range(0,5)):
#     print(i)
# =============================================================================
import numpy as np
lst = [1,2,3,4]
lst = np.array(lst)
print(lst.reshape(-1,4))