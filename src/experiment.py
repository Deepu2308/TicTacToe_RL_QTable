# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:54:23 2021

@author: deepu
"""
from multiprocessing import Pool
import itertools
import logging
from datetime import datetime

#import custom modules
from models import sarsa

if __name__ == '__main__':
    
    
    grid_steps      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    grid            = list(itertools.product(grid_steps,grid_steps,grid_steps))
    
    #setup logger
    logging.basicConfig(filename='src/logs/sarsa.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
    
    #performe sarsa training 7 at a time
    logging.info(f"Starting experiments at {datetime.now()}")
    logging.info(f"Number of experiments : {len(grid)}")
    p = Pool(7)
    p.map(sarsa,grid)
    logging.info(f"Experiments completed at {datetime.now()}")