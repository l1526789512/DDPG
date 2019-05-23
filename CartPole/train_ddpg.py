from DDPGAgent import DDPG
import gym
import numpy as np
import time

RENDER = False
ENV_NAME = 'CartPole-v1'
MAX_EPISODES = 2000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000

#training
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = 1
#a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound=None)

var = 2 #control exploration
t1 = time.time()
for episode in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a_prob = ddpg.choose_action(s)
        #print(a_prob)
        a_prob = np.clip(np.random.normal(a_prob, var), -1, 1)
        #print(a_prob)
        a=1
        if a_prob<0:
            a=0
        #print(a)
        #a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)
        
        r = -10 if done else 1

        ddpg.store_transition(s, a_prob, r, s_, done)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1 or done:
            #print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
            
    if episode % 100 == 0:
        total_reward = 0
        for i in range(10):
            state = env.reset()
            for j in range(MAX_EP_STEPS):
                #env.render()
                action_prob = ddpg.choose_action(state) # direct action for test
                action=1
                if action_prob<0:
                    action=0
                state,reward,done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward/10
        print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
print('Running time: ', time.time() - t1)

