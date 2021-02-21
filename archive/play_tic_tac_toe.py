# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:29:04 2018

@author: dunnikrishnan
"""

import numpy as np
from itertools import combinations_with_replacement
import pandas as pd
from random import sample,uniform
import itertools


#edit path variables here!!!!!
path_qTable_player1 = r'.../player_1_policy_v2.csv'
path_qTable_player2 = r'.../player_2_policy_v2.csv' 

mat = np.array([[0,0,1],[0,1,0],[1,0,0]])


global N , gamma
N       = mat.shape[0]
#gamma   = .95
gamma   = 1
alpha   =.7

def feedback(mat):
    
    winner = 0
    game_over = 0
    
    col_sum = mat.sum(axis = 0)
    row_sum = mat.sum(axis = 1) 
    dia_sum = sum(mat.diagonal())
    rev_dia = sum(mat[::-1].diagonal())
    
    
    l = list(itertools.chain(*[col_sum, row_sum ]))
    l = l + [dia_sum] + [rev_dia]
    if   any( [j == N for j in l]):
        winner,game_over = 1,1
        return  winner,game_over
    elif any( [j == -N for j in l]):
        winner,game_over = -1,1
        return  winner,game_over
    
    if np.count_nonzero(mat) == N*N:
        return 0,1
    else:
        return 0,0
    
def create_q_table(N):
    
    l = list(combinations_with_replacement([0,1,2],2))
    
    l_rev = [i[::-1] for i in l if i[0] != i[1]]
    
    l = [ '_'.join([str(i[0]) , str(i[1])]) for i in l + l_rev]
    
    q_table = pd.DataFrame( columns = l)  
    
    return q_table

def choose_action(state,mat,q_table):
    
    """only cells that are not already filled"""
    options = [i for i in q_table.columns if mat[int(i[0]),int(i[2])] == 0]    
   
    if state not in q_table.index:
        q_table.loc[state] = 0
        return q_table ,sample(options, 1)[0]        
       
    else:
        all_zeroes = all([q_table.loc[state,i]== 0 for i in options])
        if  uniform(0,1) > gamma or all_zeroes:
            return q_table , sample(options, 1)[0]

        else :        
            values = [q_table.loc[state,i] for i in options]
            return q_table , options[values.index(max(values))]
   
def update_env(mat,player,action):
    
    action                      = [int(i) for i in action.split('_')]    
    mat[action[0] , action[1]]  = player
    
    return mat

def update_q_table(q_table, prev_state,  prev_action, state):    
    
    if state not in q_table.index:
        q_table.loc[state] = 0
        
    q_table.loc[prev_state,  prev_action] = q_table.loc[prev_state,
                   prev_action]*alpha + (1-alpha)*max(q_table.loc[state])
    
    return q_table

def decode_state(state):
    state_ = state.split('|')[1:]
    m = np.zeros((N,N))
    for i in state_:
        a,b = int(i[0]) , int(i[2])
        if i[1] == 'x':
            m[a,b] =1
        elif i[1] == 'o':
            m[a,b] =-1
            
       
    return m

def encode_state(mat):
    
    X,O = [],[]
    
    for i in range(N):
        for j in range(N):
            
            if mat[i,j] == 1:
                X.append((i,j))
            elif mat[i,j] == -1:
                O.append((i,j))

    X.sort(key = lambda x: (x[0],x[1]) )
    O.sort(key = lambda x: (x[0],x[1]) )

    X_ = ['x'.join([str(x[0]),str(x[1])]) for x in X]
    O_ = ['o'.join([str(o[0]),str(o[1])]) for o in O]
    
    return 'S|' + '|'.join(X_ + O_)
    
def display_mat(mat):
    print()
    for i,j in pd.DataFrame(mat).iterrows():
        print('|'.join([str(k) for k in j]).replace('-1','O').replace('1','X').replace('0','_'))
        

player_1 = {'q_table' : create_q_table(N) , 'history' : [] }
player_2 = {'q_table' : create_q_table(N) , 'history' : [] }
state_dict = {}

game_info = []

#open if condition if you want to retrain, else load game from memory
if False:
    for game in range(300000):
        print("Game Number: " ,game)
        mat = np.zeros((N,N),dtype = int)
        state = encode_state(mat)
        game_over = False
        player_1['history'] = []
        player_2['history'] = []
        
        if len(game_info) + 1 % 5000 == 0:
           print(np.mean(game_info[-500:]))
        print("Wins:{}| Losses:{}| Draws:{}".format(game_info.count(1) , 
                              game_info.count(-1) , game_info.count(0)))
        
        while not game_over:   
            
            """player 1 moves"""
            
            """choose action"""
            
            player_1['q_table'] , action = choose_action(state,mat,player_1['q_table'])
            prev_state_1 ,  prev_action_1 = state,action
            
            """update env and redefine state"""
            mat = update_env(mat , 1 , action)
            state = encode_state(mat)
         
            if state not in state_dict.keys():
                state_dict[state] = 1
            else:
                state_dict[state] += 1
            
            """get feedback"""
            winner,game_over = feedback(mat)
            player_1['history'].append((prev_state_1 , prev_action_1 , winner))
            #player_2['history'].append((prev_state_2 , prev_action_2 , -winner))
            
            """update q table"""
            if winner == 1:
                player_1['q_table'].loc[prev_state_1 ,  prev_action_1] = 1
                player_2['q_table'].loc[prev_state_2 ,  prev_action_2] = -1
                
                player_1['history'] = player_1['history'][::-1]
                player_2['history'] = player_2['history'][::-1]
                
                for ind,saf in enumerate(player_1['history'][1:]):
                    player_1['q_table'] = update_q_table(player_1['q_table'] ,
                            saf[0] , saf[1] , player_1['history'][ind][0] )
                    
                for ind,saf in enumerate(player_2['history'][1:]):
                    player_2['q_table'] = update_q_table(player_2['q_table'] ,
                            saf[0] , saf[1] , player_2['history'][ind][0] ) 
                    
                    
                    
                    
            player_1['q_table']  =  update_q_table(player_1['q_table'] , 
                                prev_state_1 ,  prev_action_1 , state    )
            if  game_over:   
                display_mat(mat)
                game_info.append(winner)
                print("Game Over")  
                if winner == 1:
                    
                    print("Player 1 wins")
                break
            
            
            """player 2 moves"""
            
            """choose action"""
            player_2['q_table'] , action = choose_action(state,mat,player_2['q_table'])
            prev_state_2 ,  prev_action_2 = state,action
            
            """update env and redefine state"""
            mat = update_env(mat , -1 , action)
            state = encode_state(mat)
            
            """get feedback"""
            winner,game_over = feedback(mat)
            player_2['history'].append((prev_state_2 , prev_action_2 , -winner))
            #player_1['history'].append((prev_state_1 , prev_action_1 , winner))
            
            """update q table"""
            if winner == -1:
                player_2['q_table'].loc[prev_state_2 ,  prev_action_2] = 1
                player_1['q_table'].loc[prev_state_1 ,  prev_action_1] = -1
                
                player_2['history'] = player_2['history'][::-1]
                player_1['history'] = player_1['history'][::-1]
                
                for ind,saf in enumerate(player_2['history'][1:]):
                    player_2['q_table'] = update_q_table(player_2['q_table'] ,
                            saf[0] , saf[1] , player_2['history'][ind][0] )
                    
                for ind,saf in enumerate(player_1['history'][1:]):
                    player_1['q_table'] = update_q_table(player_1['q_table'] ,
                            saf[0] , saf[1] , player_1['history'][ind][0] )
        
                    
                    
                    
            player_2['q_table']  =  update_q_table(player_2['q_table'] , prev_state_2 ,
                                    prev_action_2 , state    )       
            if  game_over:
                game_info.append(winner)
                display_mat(mat)
                print("Game Over") 
                if winner == -1:                
                    print("Player 2 wins")    
                break
            
else:
        
    player_1['q_table']  = pd.read_csv(path_qTable_player1 ,index_col= 0)
    player_2['q_table']  = pd.read_csv(path_qTable_player2 ,index_col= 0)
    
    
            
            
#df = player_1['q_table']        
#df_ = player_2['q_table']   

#df.to_csv(r'C:\Users\dunnikrishnan\Desktop\Projects\Tutorials\Reinforcement Learning\tic tak toe results/player_1_policy_v2.csv')
#df_.to_csv(r'C:\Users\dunnikrishnan\Desktop\Projects\Tutorials\Reinforcement Learning\tic tak toe results/player_2_policy_v2.csv')


def play_against_player_1():
    mat = np.zeros((N,N),dtype = int)
    state = encode_state(mat)
    game_over = False
    while not game_over:
        player_1['q_table'] , action = choose_action(state,mat,player_1['q_table'])
        print("Computers turn")
        """update env and redefine state"""
        mat = update_env(mat , 1 , action)
        state = encode_state(mat)
        display_mat(mat)
        if state not in state_dict.keys():
            state_dict[state] = 1
        else:
            state_dict[state] += 1
        
        """get feedback"""
        winner,game_over = feedback(mat)
        
        """update q table"""
        if game_over :
            
            if winner == 0:
                print("Game Drawn")
            elif winner == -1:
                print("You won")
            else:
                print("Computer won")
            break           
                
         
        action = input("Enter coordinates eg. 1_2 : ")
        print("You entered:\t", action[0] ,"," , action[2])
        mat = update_env(mat , -1 , action)
        state = encode_state(mat)
        display_mat(mat)
        winner,game_over = feedback(mat)
        if game_over :
            
            if winner == 0:
                print("Game Drawn")
            elif winner == -1:
                print("You won")
            else:
                print("Computer won")
            break
        
def play_against_player_2():
    mat = np.zeros((N,N),dtype = int)
    state = encode_state(mat)
    game_over = False
    while not game_over:
        display_mat(mat)
        action = input("Enter coordinates eg. 1_2 : ")
        print("You entered:\t", action[0] ,"," , action[2])
        mat = update_env(mat , 1 , action)
        state = encode_state(mat)
        display_mat(mat)
        winner,game_over = feedback(mat)
        
        if game_over :
            
            if winner == 0:
                print("Game Drawn")
            elif winner == 1:
                print("You won")
            else:
                print("Computer won")
            break           

        
        print("Computers turn")
        player_2['q_table'] , action = choose_action(state,mat,player_2['q_table'])
        """update env and redefine state"""
        mat = update_env(mat , -1 , action)
        state = encode_state(mat)
        #display_mat(mat)
        """get feedback"""
        winner,game_over = feedback(mat)
        if game_over :
            
            if winner == 0:
                print("Game Drawn")
            elif winner == 1:
                print("You won")
            else:
                print("Computer won")
            break           
 
