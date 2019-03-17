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
        self.holes = []
        y = 0
        for j in envDesc:
            x = 0
            for i in j:
                if(i == b'S'):
                    self.start = (x, y)
                if(i == b'H'):
                    self.holes.append((x, y))
                if(i == b'G'):
                    self.goal = (x,y)
                x+=1
            y+=1
        self.width = x
        self.height = y

    def inbounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y <self.height

    def notBlocked(self, id):
        return id not in self.holes


    def neighbours(self, id):
        (x, y) = id
        results = [(x, y - 1), (x+1, y), (x, y + 1), (x - 1, y)]
        results = filter(self.inbounds, results)
        results = filter(self.notBlocked, results)
        return results

    def printGrid(self):
        print(" ", end = " ")
        for n in range(self.width - 1):
            print(n, end = " ")
        print("\n")
        for j in range(self.height - 1):
            print(j, end = " ")
            for i in range(self.width - 1):
                if(i,j) in self.holes:
                    print("H", end = " ")
                elif (i,j) == self.start:
                    print("S", end = " ")
                elif (i,j) == self.goal:
                        print("G", end = " ")
                else:
                    print("F", end = " ")
            print("\n")





def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2) + abs(y1-y2)

def aStar(grid):
    #initialize open list
    openSet = PriorityQueue()
    openSet.put(grid.start, 0)
    cameFrom = {}
    costTo = {}
    cameFrom[grid.start] = None
    costTo[grid.start] = 0

    while not openSet.empty():
        current = openSet.get()

        if current == grid.goal:
            break

        for next in grid.neighbours(current):
            newCost = costTo[current] + 1
            if next not in costTo or newCost < costTo[next]:
                costTo[next] = newCost
                priority = newCost + heuristic(grid.goal, next)
                openSet.put(next, priority)
                cameFrom[next] = current



    a = cameFrom[grid.goal]
    b = grid.goal
    steps = []
    # grid.printGrid()
    while not a == None:
        steps.append(stepTo(b, a))
        # print(b)
        b = a
        a  = cameFrom[a]

    # steps.append(stepTo(b, grid.start))
    steps.reverse()
    # print(steps)
    # for i in steps:
    #     if(i == 0):
    #         print("Left")
    #     if(i == 1):
    #         print("Down")
    #     if (i==2):
    #         print("Right")
    #     if (i == 3):
    #         print("Up")

    return steps



def stepTo(to, fr):

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    (xt, yt) = to
    (xf, yf) = fr
    if(yt < yf and xt == xf):
        return UP
    if(xt > xf and yt == yf):
        return RIGHT
    if(yt > yf and xt == xf):
        return DOWN
    if(xt < xf and yt == yf):
        return LEFT





def main(p_id):
    # Setup the parameters for the specific problem (you can change all of these if you want to)
    problem_id = int(p_id)    # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
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

    # Create a representation of the state space for use with AIMA A-star
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    # print(state_goal_id)
    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)
    #setup vars for logfile
    f= open("out_AStar_{}.txt".format(problem_id) ,"w+")
    successes = 0
    failures = 0
    ####
    for e in range(1): # iterate over episodes
        observation = env.reset() # reset the state of the env to the starting state
        steps = aStar(g)
        for iter in range(max_iter_per_episode):
            # env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line

            action = steps[iter]
            # print(action)
            observation, reward, done, info = env.step(action) # observe what happends when you take the action

            #         # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner

            # Check if we are done and monitor rewards etc...
            if(done and reward==reward_hole):
            # env.render()
                print("Failure")
                failures += 1
                f.write("e,iter,reward,done = " + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done) + "\n")
                # f.write("We have reached a hole :-( [we can't move so stop trying; just give up]\n")
                break

            if (done and reward == +1.0):
                # env.render()
                successes += 1
                print("Success")
                f.write("e,iter,reward,done = " + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done) + "\n")
                # f.write("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]\n")
                break


    print("Successes: ", successes)
    print("Failures: ", failures)

if __name__ == "__main__":
    main(sys.argv[1])
