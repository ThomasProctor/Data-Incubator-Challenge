
# coding: utf-8

# In[1]:

from __future__ import division
from itertools import product
from primefac import primefac
import numpy as np
import pandas as pd


# In[2]:

def all_rolls(n_dice, n_sides=6):
    '''generates all possible rolls of n_dice, ignoring order.
    Assumes sides are numbered 1 to n_sides'''
    return list(product(range(1, n_sides + 1), repeat=n_dice))
    
    
    


# In[3]:

def gen_roll_sum_df(n_dice, n_sides=6):
    rolls = all_rolls(n_dice, n_sides=n_sides)
    roll_array = np.array(rolls)
    roll_sum = roll_array.sum(axis=1)
    roll_prod = roll_array.prod(axis=1)
    return pd.DataFrame({'roll':rolls, 'roll_sum':roll_sum,
                         'roll_prod':roll_prod})


# In[4]:

n_dice_8 = gen_roll_sum_df(8)


# In[15]:

print('Mean product is: ')
mean = n_dice_8[n_dice_8['roll_sum'] == 24]['roll_prod'].mean()
print('{:5.10f}'.format(mean))


# In[16]:

print('Product Standard deviation is:')
std = n_dice_8[n_dice_8['roll_sum'] == 24]['roll_prod'].std()
print('{:5.10f}'.format(std))


# In[ ]:



