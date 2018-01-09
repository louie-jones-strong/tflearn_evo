import random
import numpy as np


def split( old_weights , new_size ):
    new_weights = []
    for loop in range(new_size-1):
        new_weights += [old_weights]

    new_weights = mutation(new_weights,1,4)

    new_weights += [old_weights]
    return new_weights

def mutation( weights , mutation_rate , mutation_amount ):

    if type(weights) is np.float32 or type(weights) is int or type(weights) is float or type(weights) is np.float64:

        if mutation_rate >= random.randint(0,100)/100:

            amount = random.randint(-10000,10000)/(10**mutation_amount)
            weights = weights + amount
    else:
        weights = list(map(lambda x: mutation(x,mutation_rate,mutation_amount) , weights ))


    return weights

def fitness_cal( error ):

    if np.amin(error) < 0:
        selection_chance = error + (np.abs(np.amin(error)))+1
    else:
        selection_chance = error

    selection_chance = list(map(lambda x: x**5, selection_chance))

    selection_chance = selection_chance / np.amax(selection_chance)

    selection_chance = selection_chance / np.sum(selection_chance)

    selection_chance = list(map(lambda X: float(X), selection_chance))

    return selection_chance

def kill( selection_chance , old_weights ):

    new_weights = []
    new_selection_chance = []
    temp_array = list( range(len(old_weights)) )

    for loop in range( int( len(old_weights)/2 ) ):
        temp = np.random.choice(temp_array, p = selection_chance)
        new_weights += [ old_weights[temp] ]
        new_selection_chance += [ selection_chance[temp] ]
    
    new_selection_chance = new_selection_chance / np.sum(new_selection_chance)
    return new_selection_chance , new_weights

def breed( selection_chance , old_weights ):

    temp_array = list(range( len(old_weights) ))
    new_weights = old_weights

    for loop in range(len(old_weights)):
        temp  = np.random.choice( temp_array , p=selection_chance )
        temp2 = np.random.choice( temp_array , p=selection_chance )


        DNA_1 = old_weights[temp]
        DNA_2 = old_weights[temp2]
        temp_chance = [selection_chance[temp],selection_chance[temp2]]
        temp_chance = temp_chance / np.sum(temp_chance)
        
        
        temp = join_weights( [DNA_1,DNA_2] , temp_chance )
        
        new_weights += [ mutation( temp , 0.8 , 5 ) ]

    return new_weights #remake

def join_weights( DNA , chance_array ):
    DNA_1 = DNA[0]
    DNA_2 = DNA[1]

    if type(DNA_1) is np.float32 or type(DNA_1) is int or type(DNA_1) is float or type(DNA_1) is np.float64:
        if np.random.choice( [True,False] , p=chance_array ):
            weights = DNA_1
        else:
            weights = DNA_2
    else:
        weights = list(map( lambda X: join_weights( [DNA_2[X],DNA_1[X]] , chance_array ) , list(range(len(DNA_1))) ))

    return weights