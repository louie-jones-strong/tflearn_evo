import os
import pygame

def tag_edit( tag , line , type="str" ): 
    lenght_of_tag = len(tag)
    temp = str(line[lenght_of_tag:])
    if type == "str":
        text_on_line = str(temp)

    elif type == "int":
        text_on_line = int(temp)

    elif type == "float":
        text_on_line = float(temp)

    elif type == "bool":
        if temp == "True" or temp == "True\n":
            text_on_line = True

        elif temp == "False" or temp == "False\n":
            text_on_line = False

        else:
            text_on_line = bool(int(temp))

    return text_on_line

def folder_picker( address, auto_picked=None ):
    temp = os.listdir(address)

    if auto_picked == None:
        for loop in range(len(temp)):
            print("["+str(loop)+"] = " + str(temp[loop]))
        user_input = int(input("pick the file: "))

    else:
        user_input = auto_picked

    output_address = address + temp[user_input]

    return output_address

def sound_setup( address ):
    pygame.mixer.init()
    pygame.mixer.music.load(address)
    return

def play_sound( address=None ):
    if address == None:
        pygame.mixer.music.play()
    else:
        pygame.mixer.music.load(address)
        pygame.mixer.music.play()

    return