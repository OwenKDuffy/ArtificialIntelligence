"""
University of Glasgow
Artificial Intelligence 2018-2019
Assessed Exercise

Reinforcement learning agent for the CUSTOM Open AI Gym problem used in AI (H) '18-'19


"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
# print("Working dir:"+os.getcwd())
# print("Python version:"+sys.version)


def main(p_id):
    # Setup the parameters for the specific problem (you can change all of these if you want to)
    problem_id = int(p_id)        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
    reward_hole = -1.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent


    # Generate the specific problem
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False,   reward_hole=reward_hole)

    #q-learning variables
    epsilon = 0.5                   # degree of randomness, I found a lower rate leads to better results in the long term
    max_episodes = 2000             # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_iter_per_episode = 500      # you decide how many iterations/actions can be executed per episode

    lr_rate = 0.81
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #make a random move
        else:
            action = np.argmax(Q[state, :])
        return action

    def learn(state, state2, reward, action):
        predict = Q[state, action]
        target = reward + gamma * np.max(Q[state2, :])
        Q[state, action] = Q[state, action] + lr_rate * (target - predict)

    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)

    #setup vars for logfile
    f= open("out_RL_{}.txt".format(problem_id) ,"w+")
    successes = 0
    failures = 0
    ####
    for e in range(max_episodes): # iterate over episodes
        state = env.reset() # reset the state of the env to the starting state

        for iter in range(max_iter_per_episode):
    #         env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line
            action = choose_action(state) # your agent goes here (the current agent takes random actions)

            observation, reward, done, info = env.step(action) # observe what happends when you take the action

            learn(state, observation, reward, action)

            state = observation
    #         # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner

            # Check if we are done and monitor rewards etc...
            if(done and reward==reward_hole):
                # env.render()
                # print("Failure")
                failures += 1
                f.write("e,iter,reward,done = " + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done) + "\n")
                # f.write("We have reached a hole :-( [we can't move so stop trying; just give up]\n")
                break

            if (done and reward == +1.0):
                # env.render()
                successes += 1
                # print("Success")
                f.write("e,iter,reward,done = " + str(e) + " " + str(iter)+ " " + str(reward)+ " " + str(done) + "\n")
                # f.write("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]\n")
                break


    f.write("Successes: " + str(successes))
    f.write("\n")
    f.write("Failures: " + str(failures))
    successRate = successes / max_episodes * 100
    dict = {"Success": successes,
            "Failures": failures,
            "Episodes": max_episodes,
            "SuccessRate": successRate}
    return dict

if __name__ == "__main__":
    main(sys.argv[1])
