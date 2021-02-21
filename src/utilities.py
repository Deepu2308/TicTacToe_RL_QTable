# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:00:52 2021

@author: deepu
"""
#import libaries
from random import choice, uniform
import numpy as np


#define policy
def e_greedy(state,env,q_value,gamma = .9,  inference = False):
    
    #dont explore during inference
    gamma = 1.01 if inference else gamma
    
    
    if uniform(0,1) > gamma: #explore
        action = random_player(env)
        
    else:                  #exploit    
        if state not in q_value.keys(): #create state in q table
            q_value[state] = np.random.uniform(-.1,.1,9)
            
        if inference:
            
            #action = np.argmax([i if (i in env.available_actions()) else -np.inf for i in q_value[state]])
            action = np.argmax([v if i in env.available_actions() else -np.inf \
                                for i,v in enumerate(q_value[state])])     
                
            #print(q_value[state])
        else:
            action = np.argmax(q_value[state])
        
    return action,q_value

def random_player(env):
    return choice(env.available_actions())

def play_against_random(env, q_value, n_episodes = 100,
                        play_as = 'O', render = False, self_play = False):
    """ 
    'O' is player 1
    
    #to play as X
    play_as,running_reward = play_against_random(env, q_value, n_episodes = 1000, play_as = 'X')
    
    #to play as O
    play_as,running_reward = play_against_random(env, q_value, n_episodes = 1000, play_as = 'O')
    """
    
    assert play_as in ['X','O'], "Player should be X or O"
    
    
    running_reward = []
    
    for episode  in range(n_episodes):
        
        #start episode
        state = env.reset()
        done  = False
        
        while not done:
            
            if  play_as == state[1] :
                #print("q learner")
                action  = e_greedy(state,env,q_value, inference = True)[0]
                
            else:
                if self_play:
                    action  = e_greedy(state,env,q_value, inference = True)[0]
                else:
                    action = random_player(env)
                
            state,reward,done, _ = env.step(action)
            
        if render:
            env.render()
            print(reward, "\n\n")
        running_reward.append(reward)
    
    if play_as == 'X':
        running_reward = [-i for i in running_reward] 
    
    performance = np.mean(running_reward)
    
    won  = sum([1 if i == 1 else 0 for i in running_reward])
    lost = sum([1 if i == -1 else 0 for i in running_reward])
    draw = sum([1 if i == 0 else 0 for i in running_reward])
    
    #print(f"Player : {play_as} | Performance : {performance} | Won: {won} | Lost: {lost} | Draw: {draw} | Total : {n_episodes}")
    
    return (won,lost,draw)
    
    