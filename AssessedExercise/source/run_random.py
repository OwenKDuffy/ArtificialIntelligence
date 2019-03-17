"""
  University of Glasgow
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic random agent for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)

def main(p_id):
    # Setup the parameters for the specific problem (you can change all of these if you want to)
    problem_id = int(p_id)        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
    reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

    max_episodes = 2000   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_iter_per_episode = 500 # you decide how many iterations/actions can be executed per episode

    # Generate the specific problem
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)

    # Let's visualize the problem/env
    # print(env.desc)

    # Create a representation of the state space for use with AIMA A-star
    # state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)

    #setup vars for logfile
    f= open("out_Random_{}.txt".format(problem_id) ,"w+")
    successes = 0
    failures = 0
    ####
    for e in range(max_episodes): # iterate over episodes
        observation = env.reset() # reset the state of the env to the starting state

        for iter in range(max_iter_per_episode):
          #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line
          action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
          print(action)
          observation, reward, done, info = env.step(action) # observe what happends when you take the action

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
