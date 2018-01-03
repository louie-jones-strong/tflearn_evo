import tflearn
import numpy as np

def accuracy_cal( inputs , targets , model , batch_size , decimal_places = 1):
    outputs = run_inputs( inputs , model , batch_size)
    outputs = np.around( outputs , decimals = decimal_places )

    fitness = 0
    for loop in range(len(inputs)):
        if np.array_equal( outputs[loop] , targets[loop] ) :
            fitness += 1


    return fitness

def run_inputs( inputs , model , batch_size):

    small = 0
    large = len(inputs)%batch_size
    temp = inputs[ small : large ]
    outputs = model.predict(temp).tolist()


    for loop in range(0,int(len(inputs)/batch_size)):
        small = large
        large += batch_size
        temp = inputs[ small : large ]

        temp = model.predict(temp).tolist()
        outputs = outputs + temp
    return outputs

def model_maker( input_shape , structre_array , batch_size=20 , lr=0.01 , tensorboard_level=0 , checkpoint_on=False , checkpoint_num=1 , optimizer="adam" ):
    tflearn.config.init_graph (gpu_memory_fraction=0.95, soft_placement=True)
    model_ID = ""

    #makes the input layer
    if len(input_shape) == 1:
        network = tflearn.input_data(shape=[None, input_shape[0] ], name='input')
    elif len(input_shape) == 2:
        network = tflearn.input_data(shape=[None, input_shape[0] , input_shape[1] ], name='input')
    elif len(input_shape) == 3:
        network = tflearn.input_data(shape=[None, input_shape[0] , input_shape[1] , input_shape[2] ], name='input')
    else:
        network = tflearn.input_data(shape=[None, input_shape[0] , input_shape[1] , input_shape[2] , input_shape[3] ], name='input')

    #makes the network with recursion on each layer
    network , model_ID = layers(network,structre_array,model_ID)

    #to pick the optimizer to use to learn the task
    if optimizer == "adam":
        model_ID += "adam"
        network = tflearn.regression(network, optimizer='adam', learning_rate = lr ,batch_size = batch_size,loss='mean_square', name='target')

    else:
        model_ID += "sgd"
        sgd = tflearn.SGD(learning_rate = lr, lr_decay = 0.01 , decay_step=50)
        network = tflearn.regression(network, optimizer=sgd, learning_rate = lr ,batch_size = batch_size,loss='mean_square', name='target')
    
    if checkpoint_on:
        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_level , tensorboard_dir='log')
    else:
        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_level, checkpoint_path='tflearn\\model.tfl',max_checkpoints=checkpoint_num, tensorboard_dir='log')

    return model , model_ID

def layers(network,structre_array,model_ID):

    if structre_array[0][0] == "conv":
        network = tflearn.conv_2d(network, structre_array[0][1], 3, activation=structre_array[0][2], regularizer="L2")
        model_ID += "c"+str(structre_array[0][1])+"."

    elif structre_array[0][0] == "ann":
        network = tflearn.fully_connected(network, structre_array[0][1] , activation=structre_array[0][2],bias = True,bias_init = "Normal")
        model_ID += "a"+str(structre_array[0][1])+"."
        if len(structre_array) > 1 and structre_array[0][3] == "True":
            network = tflearn.dropout(network, 0.8)

    elif structre_array[0][0] == "maxpool":
        network = tflearn.max_pool_2d(network, structre_array[0][1])
        model_ID += "m"+str(structre_array[0][1])+"."

    elif structre_array[0][0] == "rnn":
        network = tflearn.simple_rnn(network, structre_array[0][1] , activation=structre_array[0][2],bias = True)
        model_ID += "r"+str(structre_array[0][1])+"."

    elif  structre_array[0][0] == "lstm":
        if len(structre_array) > 2 and structre_array[0][3] == "True":
            network = tflearn.lstm(network, structre_array[0][1], activation=structre_array[0][2] , dropout=0.8 , return_seq=True)
        else:
            network = tflearn.lstm(network, structre_array[0][1], activation=structre_array[0][2] , return_seq=False)
        model_ID += "l"+str(structre_array[0][1])+"."



    if len(structre_array) > 1:
        network , model_ID = layers(network,structre_array[1:],model_ID)

    return network , model_ID

def train(X , Y , testX , testY , epochs , model , batch_size , run_ID="ID" , metrics_on=False , checkpoints_on=False):
    
    try:
        model.fit( X , Y , n_epoch=epochs , validation_set=( testX , testY ) , show_metric=metrics_on , snapshot_epoch=checkpoints_on , run_id=run_ID )
    except:
        print("batch size to big!!")
        print("batch size changed from: " + str(batch_size) + " to: " + str(int(batch_size*0.9)) )
        batch_size = int(batch_size*0.9)
        model , batch_size = train( address , X , Y , testX , testY , epochs , model , batch_size , run_ID)

    return model , batch_size