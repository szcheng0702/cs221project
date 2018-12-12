#!/usr/bin/env python2
import os

import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
import gym
from copy import deepcopy
import random
import itertools
from time import time
from copy import deepcopy
from math import sqrt, log
#from multiprocessing import Pool
import pickle
sys.path.append('/Users/sizhucheng/anaconda3/lib/python3.6/site-packages')
import pathos.pools as pp

def moving_average(v, n):
    n = min(len(v), n)
    ret = [.0]*(len(v)-n+1)
    ret[0] = float(sum(v[:n]))/n
    for i in range(len(v)-n):
        ret[i+1] = ret[i] + float(v[n+i] - v[i])/n
    return ret


def ucb(node):
    return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)


def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0


class Runner:
    def __init__(self, rec_dir, env_name, loops=300, max_depth=1000, playouts=1000):
        self.env_name = env_name
        self.dir = rec_dir+'/'+env_name

        self.loops = loops
        self.max_depth = max_depth
        self.playouts = playouts

    def print_stats(self, loop, score, maxscore,avg_time):
        sys.stdout.write('\r%3d   score:%10.3f   maxscore:%10.3f avg_time:%4.1f s' % (loop, score, maxscore,avg_time))
        sys.stdout.flush()

    # def run(self):
    #     best_rewards = []
    #     start_time = time()
    #     env = gym.make(self.env_name)
    #     #env.monitor.start(self.dir)
    #     #env = gym.wrappers.Monitor(env, self.dir)

    #     print self.env_name

    #     for loop in xrange(self.loops):
    #         env.reset()
    #         root = Node(None, None)

    #         best_actions = []
    #         best_reward = float("-inf")

    #         for _ in xrange(self.playouts):
    #             state = deepcopy(env)
    #             #del state._monitor
    #             #state.__del__()

    #             sum_reward = 0
    #             node = root
    #             terminal = False
    #             actions = []

    #             # selection
    #             while node.children:
    #                 if node.explored_children < len(node.children):
    #                     child = node.children[node.explored_children]
    #                     node.explored_children += 1
    #                     node = child
    #                 else:
    #                     node = max(node.children, key=ucb)
    #                 _, reward, terminal, _ = state.step(node.action)
    #                 sum_reward += reward
    #                 actions.append(node.action)

    #             # expansion
    #             if not terminal:
    #                 node.children = [Node(node, a) for a in combinations(state.action_space)]
    #                 random.shuffle(node.children)

    #             # playout
    #             while not terminal:
    #                 action = state.action_space.sample()
    #                 _, reward, terminal, _ = state.step(action)
    #                 sum_reward += reward
    #                 actions.append(action)

    #                 if len(actions) > self.max_depth:
    #                     sum_reward -= 100
    #                     break

    #             # remember best
    #             if best_reward < sum_reward:
    #                 best_reward = sum_reward
    #                 best_actions = actions


    #             # backpropagate
    #             while node:
    #                 node.visits += 1
    #                 node.value += sum_reward
    #                 node = node.parent

    #             # fix monitors not being garbage collected
    #             #del state._monitor
    #             #state.__del__()

    #         sum_reward = 0
    #         for action in best_actions:
    #             _, reward, terminal, _ = env.step(action)
    #             sum_reward += reward
    #             if terminal:
    #                 break

    #         best_rewards.append(sum_reward)
    #         score = max(moving_average(best_rewards, 100))
    #         maxscore=max(best_rewards)
    #         avg_time = (time()-start_time)/(loop+1)
    #         self.print_stats(loop+1, score, maxscore,avg_time)

    #     env.close()
    #     print (sum(best_rewards)/float(len(best_rewards)))
    #     with open('testfile.bin', 'wb') as f:
    #         pickle.dump(best_rewards, f)

    def mp_playout(self,playouts):
        env=self.env
        best_actions = []
        best_reward = float("-inf")

        for _ in range(playouts):
            state = deepcopy(env)
            #del state._monitor
            #state.__del__()

            sum_reward = 0
            node = self.root
            terminal = False
            actions = []

            # selection
            while node.children:
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=ucb)
                _, reward, terminal, _ = state.step(node.action)
                sum_reward += reward
                actions.append(node.action)

            # expansion
            if not terminal:
                node.children = [Node(node, a) for a in combinations(state.action_space)]
                random.shuffle(node.children)

            # playout
            while not terminal:
                action = state.action_space.sample()
                _, reward, terminal, _ = state.step(action)
                sum_reward += reward
                actions.append(action)

                if len(actions) > self.max_depth:
                    sum_reward -= 100
                    break

            # remember best
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions


            # backpropagate
            while node:
                node.visits += 1
                node.value += sum_reward
                node = node.parent


        return best_reward,best_actions


    def mp_run(self):
        best_rewards = []
        start_time = time()
        env = gym.make(self.env_name)
        #env.monitor.start(self.dir)
        #env = gym.wrappers.Monitor(env, self.dir)
       
        print (self.env_name)

        for loop in range(self.loops):
            env.reset()
            self.env=env
            self.root = Node(None, None)

            pool_num=5
            pool=pp.ProcessPool(processes=pool_num)

            chunks=[self.playouts//pool_num]*pool_num
            results=pool.map(self.mp_playout,chunks)

            best_actions=max(results)[1]

            sum_reward=0

            for action in best_actions:
                _, reward, terminal, _ = env.step(action)
                sum_reward += reward
                if terminal:
                    break

            best_rewards.append(sum_reward)
            print (best_rewards)
            score = max(moving_average(best_rewards, 100))
            maxscore=max(best_rewards)
            avg_time = (time()-start_time)/(loop+1)
            self.print_stats(loop+1, score, maxscore,avg_time)

        env.close()
        with open('testfile.bin', 'wb') as f:
            pickle.dump(best_rewards, f)



# def print_stats(loop, score, maxscore,avg_time):
#     sys.stdout.write('score:%10.3f   maxscore:%10.3f ' % (score, maxscore))
#     sys.stdout.flush()

# def combine_mpres(list_bestrewards):
#     best_rewards=[r for sublist in list_bestrewards for r in sublist]
#     score = max(moving_average(best_rewards, 100))
#     maxscore=max(best_rewards)
#     #avg_time = (time()-start_time)/(loop+1)
#     print_stats(core, maxscore,avg_time)



def main():
    # get rec_dir
    if not os.path.exists('rec'):
        os.makedirs('rec')
        next_dir = 0
    else:
        next_dir = max([int(f) for f in os.listdir('rec')+["0"] if f.isdigit()])+1
    rec_dir = 'rec/'+str(next_dir)
    os.makedirs(rec_dir)
    print ("rec_dir:{}".format(rec_dir))

    # Toy text
    #Runner(rec_dir, 'Taxi-v2',   loops=100, playouts=4000, max_depth=50).run()
    #Runner(rec_dir, 'NChain-v0', loops=100, playouts=3000, max_depth=50).run()

    Runner(rec_dir, 'Phoenix-ram-v0', loops=100, playouts=50, max_depth=9000).mp_run()

    # combine_mpres(results)

    # Algorithmic
    # Runner(rec_dir, 'Copy-v0').run()
    # Runner(rec_dir, 'RepeatCopy-v0').run()
    # Runner(rec_dir, 'DuplicatedInput-v0').run()
    # Runner(rec_dir, 'ReversedAddition-v0').run()
    # Runner(rec_dir, 'ReversedAddition3-v0').run()
    # Runner(rec_dir, 'Reverse-v0').run()


if __name__ == "__main__":
    main()