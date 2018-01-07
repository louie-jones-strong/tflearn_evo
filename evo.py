import random
import numpy as np


def split(old_weights,new_size):
    new_weights = []
    for loop in range(new_size-1):
        new_weights += [old_weights]

    new_weights = mutation(new_weights,1,1)

    new_weights += [old_weights]
    return new_weights

def mutation( weights , mutation_rate , mutation_amount ):

    if type(weights) is np.float32 or type(weights) is int or type(weights) is float or type(weights) is np.float64:

        if mutation_rate >= random.randint(0,100)/100:

            amount = random.randint(-15000,15000)/10000
            weights = weights * amount
    else:
        weights = list(map(lambda x: mutation(x,mutation_rate,mutation_amount) , weights ))


    return weights # adding mutation_amount

def fitness_cal(error):

    if np.amin(error) < 0:
        selection_chance = error + (np.abs(np.amin(error)))+1
    else:
        selection_chance = error

    selection_chance = list(map(lambda x: x**3, selection_chance))

    selection_chance = selection_chance / np.amax(selection_chance)

    selection_chance = selection_chance / np.sum(selection_chance)

    return selection_chance

def kill(selection_chance,old_weights):

    new_weights = []
    new_selection_chance = []
    temp_array = list(range(len(old_weights)))

    for loop in range(int(len(old_weights)/2)):
        temp = np.random.choice(temp_array, p = selection_chance)
        new_weights += [old_weights[temp]]
        new_selection_chance += [selection_chance[temp]]
    
    new_selection_chance = new_selection_chance / np.sum(new_selection_chance)
    return new_selection_chance , new_weights

def breed( selection_chance , old_weights ):

    temp_array = list(range( len(old_weights) ))
    new_weights = old_weights

    for loop in range(len(old_weights)):
        temp  = np.random.choice(temp_array, p = selection_chance)
        temp2 = np.random.choice(temp_array, p = selection_chance)


        DNA_1 = old_weights[temp]
        DNA_2 = old_weights[temp2]
        temp_chance = [selection_chance[temp],selection_chance[temp2]]
        temp_chance = temp_chance / np.sum(temp_chance)
        
        
        new_DNA = DNA_1
        # pick elemnt one by one from each
        for loop2 in range(len(DNA_1)):
            for loop3 in range(len(DNA_1[loop2])):
                for loop4 in range(len(DNA_1[loop2][loop3])):

                    temp  = np.random.choice([0,1], p = temp_chance)
                    if temp == 0:
                        new_DNA[loop2][loop3][loop4] = DNA_1[loop2][loop3][loop4]
                    else:
                        new_DNA[loop2][loop3][loop4] = DNA_2[loop2][loop3][loop4]

        
        new_weights += [mutation(new_DNA,0.8,1)]
    return new_weights