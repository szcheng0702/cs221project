import gym
import numpy as np

env=gym.make('Phoenix-ram-v0')

num_episodes=1000

#q_table = np.random.uniform(low=-1, high=1, size=(num_states, env.action_space.n))

def bins(clip_min, clip_max, num):
	return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(low, high, observation):

	state_array = observation
	digitized=np.zeros(state_array.shape)

	# for i in state_array.shape[0]:
	   #  digitized[i] = np.digitize(state_array[i], bins=bins(low[i], high[i], 4))
	 
	return sum(digitized)

#baseline get action
def get_action(low,high,observation,epsilon=0.5):
	curr_state=digitize_state(low,high,observation)
	if  epsilon <= np.random.uniform(0, 1):		
		next_action = np.digitize(curr_state,bins(sum(low),sum(high),8))
	else:
		next_action = np.random.choice(np.arange(8))  

	next_state=curr_state+next_action*(sum(high)-sum(low))
	return next_state,next_action


low=env.observation_space.low
high=env.observation_space.high

max_number_of_steps=200
rewards=[]
#baseline, not learning anything
for episode in range(num_episodes):
	env.reset()
	action = env.action_space.sample()
	total_reward=0
	for t in range(max_number_of_steps):
		env.render()
		observation, reward, done, info = env.step(action)
		total_reward+=reward  
		_,action= get_action(low,high,observation)  
		if done or t==max_number_of_steps-1:
			print ('total reward is {}'.format(total_reward))
			print('Episode %d DONE!' % episode)
			rewards.append(total_reward)
			break
print (np.mean(np.asarray(rewards)))

env.close()
# env.reset()
# for _ in range(1000):
# 	env.render()
# 	env.step(env.action_space.sample()) # take a random action


