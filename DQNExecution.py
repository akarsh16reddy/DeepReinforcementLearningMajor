from VM import VM,getResourcePool
from read_dataset import get0TimeStepJobs
from Job import Job
from statistics import mean
import matplotlib.pyplot as plt
import random
from collections import deque
import tensorflow as tf
import numpy as np
from tqdm import tqdm,trange
import time
from csv import writer
import sys


REPLAY_MEMORY_SIZE = 800
MIN_REPLAY_MEMORY_SIZE = 500
SIZE_OF_MINI_BATCH = 30
LEARNING_RATE_OVERALL=0.01

tow = 500
f = 1
epsilon = 0.9
epsilon_decay = 0.002
epsilon_lower_bound = 0.1



episode=0

target_update_counter=0
    
class DQNAgent:
    def __init__(self):

       # Main model
       self.evaluation_network = self.create_model()
       
       #target_network
       self.target_network = self.create_model()
       
       self.target_network.set_weights(self.evaluation_network.get_weights())
       self.SIZE_OF_MINI_BATCH = 30
       self.target_update_counter = 0
       self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
       self.gamma = 0.9
       self.epsilon= 0.01
       self.epsilon_max = 0.9
       self.learn_step_counter=0
       self.UPDATE_AFTER_DECISION_EPISODES=50
       self.epsilon_increment=0.002
       
       self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)
       
    def save_model(self):
        self.evaluation_network.save('evaluation_model_2')
        self.target_network.save('target_network_2')
       
    def create_model(self):
        model = tf.keras.models.Sequential()
        w_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
        b_initializer = tf.keras.initializers.Constant(value=0.1)
        model.add(tf.keras.layers.Dense(20,input_shape=(12,),activation='relu',kernel_initializer=w_initializer,bias_initializer=b_initializer))
        model.add(tf.keras.layers.Dense(10,kernel_initializer=w_initializer,bias_initializer=b_initializer))
        opt=tf.keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss=self.squared_difference_loss_function, optimizer=opt, metrics=['accuracy'])
        return model
    
    def choose_action(self,state):
        pro = np.random.uniform()
        if pro<self.epsilon:
            actions_value = self.evalutaion_network.predict(np.array(state))
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,10)
        return action
    
    def choose_best_action(self, state):
        actions_value = self.evaluation_network(np.array(state))
        action = np.argmax(actions_value)
        return action

    def update_replay_memory(self, current_state, action, reward, next_state):
        one_hot_action = np.zeros(10)
        one_hot_action[action] = 1
        self.replay_memory.append((current_state, one_hot_action, reward, next_state))
            
    def get_qs(self, state):
        state = np.array(state)
        return self.evaluation_network.predict(np.array(state).reshape(-1, *state.shape))
    
    def squared_difference_loss_function(self,y_true,y_pred):
        squared_difference = tf.math.square(y_true - y_pred)
        return tf.math.reduce_mean(squared_difference)
    
# =============================================================================
#     def calculate_loss(self):
#         self.action_input = feed_dict[action_input]
#         self.q_target = feed_dict[q_target]
#         q_evaluate = tf.reduce_sum(input_tensor=tf.math.multiply(self.q_eval, self.action_input), axis=1)
#         loss = tf.math.reduce_mean(input_tensor=tf.math.squared_difference(self.q_target, q_evaluate))
# =============================================================================
        
    def train(self):
        if self.learn_step_counter%self.UPDATE_AFTER_DECISION_EPISODES==0:
            self.target_network.set_weights(self.evaluation_network.get_weights())
            
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.SIZE_OF_MINI_BATCH)
        
        state_batch = [np.array(data[0]) for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        
        q_next_batch = self.target_network.predict(next_state_batch)
        q_real_batch=[]
        for i in range(self.SIZE_OF_MINI_BATCH):
            q_real_batch.append(minibatch[i][2] + self.gamma * np.max(q_next_batch[i]))
        
        self.evaluation_network.fit(state_batch,q_real_batch,epochs=1,verbose=0)
        
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max

        self.learn_step_counter += 1
                
# =============================================================================
#         # Get current states from minibatch, then query NN model for Q values
#         current_states = np.array([transition[0] for transition in minibatch])
#         current_qs_list = self.evaluation_network.predict(current_states)
# 
#         # Get future states from minibatch, then query NN model for Q values
#         # When using target network, query it, otherwise main network should be queried
#         new_current_states = np.array([transition[3] for transition in minibatch])
#         future_qs_list = self.target_network.predict(new_current_states)
# 
#         X = []
#         y = []
# =============================================================================

        # Now we need to enumerate our batches
# =============================================================================
#         for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
#             max_future_q = np.max(future_qs_list[index])
#             new_q = reward + gamma * max_future_q
#                 
#             # Update Q value for given state
#             current_qs = current_qs_list[index]
#             current_qs[action] = new_q
# 
#             # And append to our training data
#             X.append(current_state)
#             y.append(current_qs)
# =============================================================================

# =============================================================================
#         for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
#             current_state = np.array(current_state)
#             new_current_state = np.array(new_current_state)
#             target = self.evaluation_network.predict(np.array(current_state).reshape(-1, *current_state.shape))
#             t = self.target_network.predict(np.array(new_current_state).reshape(-1, *new_current_state.shape))
#             target[0][action] = reward + gamma * max(t[0])
#             current_state = current_state.reshape(-1, *current_state.shape)
#             self.evaluation_network.fit(np.array(current_state), np.array(target), epochs=1, verbose=0)
#             
#         # Fit on all samples as one batch, log only on terminal state
#         #self.evaluation_network.fit(np.array(X), np.array(y),batch_size=SIZE_OF_MINI_BATCH, verbose=0, shuffle=False)
# 
#         # Update target network counter every episode
#         self.target_update_counter += 1
# 
#         # If counter reaches set value, update target network with weights of main network
#         if self.target_update_counter > UPDATE_AFTER_DECISION_EPISODES:
#             self.target_network.set_weights(self.evaluation_network.get_weights())
#             self.target_update_counter = 0
#             
#         global epsilon
#         if epsilon>epsilon_lower_bound:
#             epsilon-=epsilon_decay
# =============================================================================


# =============================================================================
# Instantizing DQN Agent 
# =============================================================================
agent = DQNAgent()

# =============================================================================
# Get resource pool -> i.id, i.VMtype, i.vComp, i.vIO
# =============================================================================
resourcePool = getResourcePool()

# =============================================================================
# VMList contains list of all VM Objects
# =============================================================================
VMList = []
for i in resourcePool:
    VMList.append(VM(i.id,i.VMtype,i.vComp,i.vIO))


# =============================================================================
# Loading entire dataset into -> pd0
# =============================================================================
pd0 = get0TimeStepJobs()


# =============================================================================
# # =============================================================================
# # Assigning first job to VM1
# # =============================================================================
# f_object1 = open('current_state.csv','a',newline='')
# writer_object1 = writer(f_object1)
# def step(current_state_f,job_f,action_f):
#     current_state_f[0]=job_f.jType
#     current_state_f[1]=job_f.jSize
#     print(VMList[action_f].getRemainingExecutionTime())
#     current_state_f[action_f+2] = VMList[action_f].getRemainingExecutionTime()
#     print(current_state_f)
#     writer_object1.writerow(current_state_f)
#     return current_state_f
#     
# # =============================================================================
# # opening csv file
# # =============================================================================
# f_object = open('responsetimes_DRL_test_afternoon_2.csv','a',newline='')
# writer_object = writer(f_object)
# # =============================================================================
# # Assigning jobs
# # =============================================================================
# 
# k=0
# job_number=0
# first_job = Job(1, 0, 1, 201)
# current_job = Job(1, 0, 1, 201)
current_state = np.zeros(12,dtype='float32')
new_state = np.zeros(12,dtype='float32')

# new_state = current_state.copy()
# action=0
# previous_state=[]
# jobLatency=0
# previous_action=-1
# 
# for i in tqdm(range(200)):
#     if i==2:
#         f_object.close()
#         f_object1.close()
#         sys.exit()
# # =============================================================================
# # jobList contains all jobs arriving at timestamp = i
# # =============================================================================
#     jobList = []
#     for row in pd0.loc[pd0['timestamp']==i].itertuples(index=True,name='Pandas'):
#             jobList.append(Job(row.id,row.timestamp,row.type,row.rounded_load))
#     
# #    jobList = jobList[:3]
#     for job in jobList:
#         if np.random.random() > epsilon:
#             action = np.argmax(agent.get_qs(current_state))
#         else:
#             action = np.random.randint(0, 10)
# 
#         new_state[0]=current_job.jType
#         new_state[1]=current_job.jSize
#         new_state[action+2] = VMList[action].getRemainingExecutionTime()
#         print(new_state)
#         writer_object1.writerow(new_state)
#         
#         VMList[action].assignJob(job)
#         executionTime=0
#         if current_job.jType==0:
#             executionTime = current_job.jSize/VMList[action].vIO
#         else:
#             executionTime = current_job.jSize/VMList[action].vComp
#         
#         assignedVmRemainingExecutionTime = VMList[action].getRemainingExecutionTime()
#         writer_object.writerow([current_job.id,current_job.arrivalT,assignedVmRemainingExecutionTime])
#             
#         jobLatency = executionTime/(assignedVmRemainingExecutionTime)
#         agent.update_replay_memory((np.array(current_state),action,jobLatency,np.array(new_state)))
#         agent.train(job_number)
#         current_job = job
#         job_number=job_number+1
#     for index,vmxx in enumerate(VMList):
#         print(index,":",vmxx.getJobList(),":",vmxx.getRemainingExecutionTime())
#         
#     for vmxx in VMList:
#         vmxx.processJobs()
#         
# agent.save_model()
# f_object.close()
# 
# 
# =============================================================================
# =============================================================================
# print()
# print(replay_memory)
# 
# f=open('f1.txt','w')
# for ele in replay_memory:
#     f.write(str(ele))
#     f.write('\n')
# f.close()
# =============================================================================
#for i in VMList:
#    i.printCurrentJobList()

# =============================================================================
#         waitingTimes = []
#         for vm in VMList:
#             waitingTimes.append(vm.getRemainingExecutionTime())
#         
#         j=0
#         temp = max(waitingTimes)
#         if job.jType == 1:
#            for m in range(0,5):
#                if temp > waitingTimes[m]:
#                    j = m
#                    temp = waitingTimes[m]
#         else:
#             for m in range(5,10):
#                 if temp > waitingTimes[m]:
#                     j = m
#                     temp = waitingTimes[m]
#         VMList[j].assignJob(job)
#     for vm in VMList:
#         vm.processJobs()
# 
# for i in VMList:
#     i.printCurrentJobList()
# 
# for vm in VMList:
#     if vm.jobListEmpty():
#         print("VM ",vm.getId(),": done")
#     else:
#         print(print("VM ",vm.getId(),": not completed"))
# =============================================================================

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