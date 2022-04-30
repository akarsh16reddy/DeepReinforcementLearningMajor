import tensorflow as tf
import numpy as np      
from collections import deque

REPLAY_MEMORY_SIZE = 800
MIN_REPLAY_MEMORY_SIZE = 500
epsilon=0.9
class DQNAgent:
    def __init__(self):

       # Main model
       self.model = self.create_model()

       # Target network
       self.target_model = self.create_model()
       self.target_model.set_weights(self.model.get_weights())

       # An array with last n steps for training
       self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

       # Used to count when to update target network with main network's weights
       self.target_update_counter = 0
       
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(20,input_shape=(12,)))
        model.add(tf.keras.layers.Dense(10))
        opt=tf.keras.optimizers.RMSprop(learning_rate=0.01)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state):
        state = np.array(state)
        return self.model.predict(np.array(state).reshape(-1, *state.shape))
    
    def printOutvalue(self):
        print(epsilon)
    

agent = DQNAgent()
model = agent.create_model()

test_state = [1,201,0,0,0,0,0,0,0,0,0,0]
test_state_np = np.array(test_state)
for i in range(0,5):
    print(agent.get_qs(test_state))
    
action = np.argmax(agent.get_qs(test_state_np))
print(action)