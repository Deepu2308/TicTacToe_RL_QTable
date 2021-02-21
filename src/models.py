# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:47:38 2021

@author: deepu
"""

#import libraries
#import src.env as environment
#from src.utilities import e_greedy,play_against_random,sq

#import custom modules
import env as environment
from   utilities import e_greedy,play_against_random

import numpy as np
import pandas as pd
import logging

def sarsa(parameters):
    
    #read parameters
    learning_rate,reward_decay,exploit_prob = parameters[0],parameters[1],parameters[2]
    
    #setup logger
    logging.basicConfig(filename='src/logs/sarsa.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
    
    #create env, qtable
    env           = environment.TicTacToeEnv()
    q_value       = {((0, 0, 0, 0, 0, 0, 0, 0, 0), 'O') : np.random.uniform(-.1,.1,9)}

    EXP_NAME = f'SARSA_LR_{learning_rate}_RD_{reward_decay}_EP_{exploit_prob}'
    perf_X  = []
    perf_O  = []
    
    for n in range(100000):#
        
        #print("Starting Game ", n)
        
        #start episode
        state = env.reset()
        done  = False

    
        #iterate till end of episode
        while not done:
            
            #show board
            #env.render()
            
            #select action
            action,q_value = e_greedy(state,env,q_value, gamma = exploit_prob)
            #print("Selected action :", action)
            
            #penalize overwrite in game board 
            if action not in env.available_actions():
                reward    = -100
                new_state = state
            else:       #step 
                
                #collect new state and reward
                new_state, reward, done, _ = env.step(action)
                
                if state[1] == 'X':
                    reward = -reward
        
            #update q values
            """        
            FYI - Q_state* is the max value acheivable from state
            
            ideally if optimal q_value is optimal
            q_value = Reward + gamma X Qvalue_next_state*
            
            since we are learning we have to set that as the target
            target_value = Reward + gamma X Qvalue_next_state
            
            and move q_value in the direction of target_value
            Qvalue[s,a] = Qvalue[s,a] + learning_rate X (target_value - current_value)
            
            rewrite that as 
            Qvalue[s,a] += learning_rate*(target_value - current_value)
            """
            
            #calculate current value
            current_value = q_value[state][action]
            
            #calculate target value
            """
            if new state is not in q_value database then initialize randomly
            """
            if new_state not in q_value.keys():
                q_value[new_state] = np.random.uniform(-.1,.1,9) 
                
            target_value  = reward + reward_decay * (-1*max(q_value[new_state]))
            
            #update
            q_value[state][action] = (1-learning_rate)*q_value[state][action] +\
                                            learning_rate*(target_value - current_value)
            
            #state has changed
            state = new_state
            
            
        #episode in now over
        #env.render()
        #print(reward)
        #print("\n\n")
        
        if (n+1)%1000 == 0:
            
            #print(f"\n\nAfter {n} itertations")
            
            #to play as O
            won,lost,draw = play_against_random(env, q_value, n_episodes = 1000, play_as = 'O', self_play = False)
            perf_O.append((won,lost,draw))
            
                        
            #to play as X
            won,lost,draw = play_against_random(env, q_value, n_episodes = 1000, play_as = 'X', self_play = False)
            perf_X.append((won,lost,draw))
           
    message = f"SARSA : {EXP_NAME} |  Won: {won} | Lost: {lost} | Draw: {draw} | Total Played : {1000}"
    print(message) 
    logging.info(message)
    
    pd.DataFrame(perf_X, columns = ['won','lost','draw']).to_csv(f'src/logs/sarsa/X_perf_{EXP_NAME}.csv')
    pd.DataFrame(perf_O, columns = ['won','lost','draw']).to_csv(f'src/logs/sarsa/O_perf_{EXP_NAME}.csv')
    pd.DataFrame(q_value).T.to_csv(f'src/logs/sarsa/qtable/QTable_{EXP_NAME}.csv')
    
if __name__ == '__main__':
    
    
    learning_rate = .5
    reward_decay  = .9
    exploit_prob  = .6
    
    sarsa([learning_rate,reward_decay,exploit_prob])