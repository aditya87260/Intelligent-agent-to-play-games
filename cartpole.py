import gym

# Create Enviroment 
env = gym.make('MountainCar-v0')
print(env.reset())

print(env.action_space)

print(env.action_space.n)

print(env.observation_space.shape[0])
"""
for t in range(1000):
    random_action = env.action_space.sample()
    env.step(random_action) #randomly move left or right
    env.render()
env.close()

#Playing game with a random strategy

for e in range(20): #Episode
    #Play 20 episodes 
    observation = env.reset()
    for t in range(50):
        env.render()
        action = env.action_space.sample()
        observation,reward,done,other_info = env.step(action)
        
        if done:
            # Game Episode is over
            print("Game Episode :{}/{} High Score :{}".format(e,20,t))
            break

env.close()
print("All 20 episodes over!")
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import tensorflow as tf


class Agent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #Discount Factor
        self.epsilon = 1.0 # Exploration Rate: How much to act randomly, 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001 
        self.model = self._create_model()
        
    
    def _create_model(self):
        #Neural Network To Approximate Q-Value function
        model = Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu')) #1st Hidden Layer
        model.add(Dense(24,activation='relu')) #2nd Hidden Layer
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) #remembering previous experiences
        
    def act(self,state):
        # Exploration vs Exploitation
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # predict reward value based upon current state
        return np.argmax(act_values[0]) #Left or Right
    
    def train(self,batch_size=32): #method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            
            if not done: #boolean 
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs=1,verbose=0) #single epoch, x =state, y = target_f, loss--> target_f - 
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self,name):
        self.model.load_weights(name)
    def save(self,name):
        self.model.save_weights(name)

#Training the DQN Agent (Deep Q-Learner)

n_episodes = 1000
output_dir = "cartpole_model/"
agent = Agent(state_size=2,action_size=3)
done = False
state_size = 2
action_size =3
batch_size = 32
agent = Agent(state_size, action_size) # initialise agent
done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state,[1,state_size])
    
    for time in range(500):
        env.render()
        action = agent.act(state) #action is 0 or 1
        next_state,reward,done,other_info = env.step(action) 
        reward = reward if not done else -10
        next_state = np.reshape(next_state,[1,state_size])
        agent.remember(state,action,reward,next_state,done)
        state = next_state
        
        if done:
            print("Game Episode :{}/{}, High Score:{},Exploration Rate:{:.2}".format(e,n_episodes,time,agent.epsilon))
            break
            
    if len(agent.memory)>batch_size:
        agent.train(batch_size)
    
    if e%50==0:
        agent.save(output_dir+"weights_"+'{:04d}'.format(e)+".hdf5")
        
env.close()