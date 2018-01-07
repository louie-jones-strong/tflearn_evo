import tflearn
import numpy as np
import os
import time
from tf_model_maker import model_maker , accuracy_cal , run_inputs , train , get_weights , set_weights , error_cal
from common_code import tag_edit , folder_picker , sound_setup , play_sound
from evo import *

class main(object):
 
    def load_data_set(self, address):
    
        #config of data-set
        file = open(address + "config.txt")
        length = tag_edit("length of data set = " , file.readline() , type = "int")
        image_type = tag_edit("image_type = " , file.readline()[:-1] , type = "bool")
        input_shape = tag_edit("input size = " , file.readline())
        input_shape = str(length) + "," + input_shape
        input_shape = input_shape.split(",")
        for loop in range(len(input_shape)):
            input_shape[loop] = int(input_shape[loop])
        input_shape = tuple(input_shape)
    
        output_shape = tag_edit("output shape = " , file.readline())
        output_shape = str(length) + "," + output_shape
        output_shape = output_shape.split(",")
        for loop in range(len(output_shape)):
            output_shape[loop] = int(output_shape[loop])
        output_shape = tuple(output_shape)
    
        batch_size = tag_edit("batch size = " , file.readline() , type = "int")
        number_of_layers = tag_edit("number of layers = " , file.readline() , type = "int")
    
        file.readline()
    
        structre_array = [file.readline()[:-1].split(",")]
        structre_array[0][1] = int(structre_array[0][1])
        for loop in range(1,number_of_layers):
            structre_array.append(file.readline()[:-1].split(","))
            structre_array[loop][1] = int(structre_array[loop][1])
    
        file.close()
        
        #read in data-set
        print("loading started:")
        time_mark = time.time()
        if image_type:
            inputs , targets = self.image_read_in(address + "data-set.txt" , length , input_shape , output_shape)
        else:
            inputs , targets = self.text_read_in(address + "data-set.txt" , length , input_shape , output_shape)
    
        print(time.time() - time_mark)
        #shuffle data-set only in the first axis
        #inputs, targets = tflearn.data_utils.shuffle (inputs,targets)
    
        print("loading data-set finished!")
        return inputs , targets , input_shape[1:] , output_shape[1:] , structre_array , batch_size
    
    def text_read_in(self, address , length , input_shape , output_shape):
        
        file = open(address)
        data_file = file.read().split("\n")
        file.close()
    
        line = data_file[0].split("!")
        inputs  = [line[0].split(",")]
        outputs = [line[1].split(",")]
    
    
        for loop in range(1,length):
            line = data_file[loop].split("!")
    

            inputs.append(line[0].split(","))
            outputs.append(line[1].split(","))
        
        del data_file
        inputs = np.asarray( inputs )
        unputs = inputs.astype(float)
        outputs = np.asarray( outputs )
        outputs = outputs.astype(float)
        inputs = np.reshape(inputs , input_shape)
        outputs = np.reshape(outputs , output_shape)
        return inputs , outputs
        
    def split_data(self, data , split_percentage ):
        training_size = int((len(data) / 100)*split_percentage)
    
        if training_size == 0:
            training_size = 1
    
        train = data[:training_size]
        if training_size == len(data) or training_size == 0:
            test  = data[:training_size]
        else:
            test  = data[training_size:]
    
        return train , test

    def save_graph(self, model , batch_size , inputs , targets , epoch):
        file = open( "graphs\\graph" + str(epoch) + ".csv","w")
        output = inputs[:1]
        for loop in range(len(inputs)):

            output = run_inputs( output , model , batch_size)

            line  = ",".join(list(map(str, output[0][:8])))
            line += ","
            line += ",".join(list(map(str, targets[loop][:8])))
            line += "\n"

            output = [list(map(lambda x: [x],output[0]))]
            output = np.asarray(output)

            file.writelines(line)

        file.close()
        print("graph saved!")
        return
    
    def setup(self):
        address = os.getcwd() + "\\info\\"
    
        file = open( address + "config.txt" , "r")
        split_percentage = tag_edit("training percentage = " , file.readline() , type = "int")
        vram_size = tag_edit("available vram(in GB) = " , file.readline() , type = "float")
        sound_on = tag_edit("sound_on = " , file.readline() , type = "bool")
        metrics_on = tag_edit("output metric = " , file.readline() , type = "bool")
        checkpoints_on = tag_edit("checkpoints = " , file.readline() , type = "bool")
        checkpoint_num = tag_edit("checkpoint_num = " , file.readline() , type = "int")
        
        file.close()
        if sound_on:
            sound_setup(address + "sounds\\coins.ogg")
    
    
        return address , split_percentage , sound_on , metrics_on , checkpoints_on , checkpoint_num

    def main(self):
        address , split_percentage , sound_on , metrics_on , checkpoints_on , checkpoint_num = self.setup()
        
        address_dataset = "info\\data-sets\\example dataset\\"
    
        num_networks = 40
        dataset_name = "dataset"
    
        inputs , targets , input_shape , output_shape , structre_array , batch_size = self.load_data_set(address_dataset)
        
        #split data in to testing and training
        train_inputs  , test_inputs  = self.split_data( inputs , split_percentage )
        train_targets , test_targets = self.split_data( targets , split_percentage )


        model , model_ID = model_maker( input_shape , structre_array , batch_size=batch_size , lr = 0.001 , tensorboard_level = 0, checkpoint_on=checkpoints_on , checkpoint_num=checkpoint_num)
        run_ID = dataset_name + "." + model_ID
        os.system("cls") 

        weights = get_weights( model , len(structre_array) )

        network_weights = split( weights , num_networks )

        total_epochs = 0
        while True:
            errors = []
            mark_start = time.time()
            for loop in range(num_networks):#loop through each network models
                model = set_weights( model , len(structre_array) , network_weights[loop] )
                errors += [0 - error_cal( inputs , targets , model , batch_size )]

            fitness = fitness_cal(errors)

            fitness , network_weights = kill(fitness,network_weights)

            network_weights = breed( fitness , network_weights )

            total_epochs += 1
            print(network_weights)
            print("errors: " + str(np.sort(np.abs(errors))))
            print("time taken: " + str(time.time() - mark_start) )
            print("epochs: " + str(total_epochs) )
            print("")
        return

if __name__ == "__main__":
    main = main()
    main.main()
