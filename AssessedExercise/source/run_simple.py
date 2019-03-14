"""
  University of Glasgow
  Artificial Intelligence 2018-2019
  Assessed Exercise

  A* style simple agent for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)
import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class Grid:
    def __init__(self, envDesc):
        self.edges= {}
        y = 0
        for j in envDesc:
            x = 0
            for i in j:
                if(i == b'F'):
                    print(x, y)
                x+=1
            y+=1

    def neighbours(self, id):
        return self.edges[id]

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)

def aStar(grid, start, goal):
    #initialize open list
    openSet = PriorityQueue()
    openSet.put(start, 0)
    cameFrom = {}
    costTo = {}
    cameFrom[start] = None
    costTo[start] = 0

    while not openSet.empty():
        current = openSet.get()

        if current == goal:
            break

        for next in grid.neighbours(current):
            newCost = costTo[current] + grid.cost(current, next)
            if next not in costTo or newCost < costTo[next]:
                costTo[next] = newCost
                priority = newCost + heuristic(goal, next)
                openSet.put(next, priority)
                cameFrom[next] = current

    return cameFrom, costTo










# Setup the parameters for the specific problem (you can change all of these if you want to)
problem_id = int(sys.argv[1])       # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = False  # should be False for A-star (deterministic search) and True for the RL agent

max_episodes = 2000   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_iter_per_episode = 500 # you decide how many iterations/actions can be executed per episode

# Generate the specific problem
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)

# Let's visualize the problem/env
# print("grid= \n")
# print(env.desc)
# env.render
g = Grid(env.desc)

# # Create a representation of the state space for use with AIMA A-star
# state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
#
# # Reset the random generator to a known state (for reproducability)
# np.random.seed(12)
#
# ####
# for e in range(max_episodes): # iterate over episodes
#     observation = env.reset() # reset the state of the env to the starting state
#
#     for iter in range(max_iter_per_episode):
#       #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line
#       action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
#       observation, reward, done, info = env.step(action) # observe what happends when you take the action
#
#       # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner
#
#       print("e,iter,reward,done =" + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done))
#
#       # Check if we are done and monitor rewards etc...
#       if(done and reward==reward_hole):
#           env.render()
#           print("We have reached a hole :-( [we can't move so stop trying; just give up]")
#           break
#
#       if (done and reward == +1.0):
#           env.render()
#           print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
#           break
