'''
Script originally written by Ben Goodrich from ALE as a direct
port of example provided in doc/examples/sharedLibraryInterfaceExample.cpp
'''

import sys
from random import randrange
from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


'''
Recording functions
'''
def make_dataset(extension='.png'):
    if len(sys.argv) < 3:
        print('Usage: %s rom_file num_games' % sys.argv[0])
        sys.exit()

    ale = ALEInterface()

    # set seed for reproducibility
    ale.setInt(b'random_seed', 123)

    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    USE_SDL = False
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            ale.setBool('sound', False) # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
        ale.setBool('display_screen', True)

    # load the ROM file
    rom_file = str.encode(sys.argv[1])
    ale.loadROM(rom_file)

    # get number of runs
    num_games = int(sys.argv[2])

    # set RGB flag
    if len(sys.argv) == 4:
        rgb = bool(sys.argv[3])

    # get the list of legal actions
    legal_actions = ale.getLegalActionSet()

    # make recording directory
    import os
    if not os.path.exists('./record/'):
        os.makedirs('./record/')
    if not os.path.exists('./record/train/'):
        os.makedirs('./record/train/')
    if not os.path.exists('record/test/'):
        os.makedirs('./record/test/')

    # initialise iteration counter
    iter = 0

    # play game
    for episode in range(num_games):
        total_reward = 0
        while not ale.game_over():
            if np.mod(iter, 2) == 0:
                screenshot_odd = ale.getScreenRGB()
            else:
                # take current screenshot as the maximum of last two
                screenshot = np.maximum(ale.getScreenRGB(), screenshot_odd)
                # pre-process image
                screenshot = __pre_process(screenshot, rgb=rgb)
                # save screenshot in appropriate directory
                __save_image(screenshot, iter/2, extension=extension)
            # select random action
            a = legal_actions[randrange(len(legal_actions))]
            # apply an action and get the resulting reward
            reward = ale.act(a)
            # increment award
            total_reward += reward
            # increment iteration counter
            iter += 1
        print('Episode %d ended with score: %d' % (episode, total_reward))
        ale.reset_game()

def __pre_process(image_array, rgb=False):
    '''
    Takes numpy array and returns processed numpy array.
    Processing extracts luminance and rescales to 84x84.
    '''
    image = Image.fromarray(image_array)
    if rgb:
        image = image.convert('RGB')
    else:
        image = image.convert('L')
    image = image.resize((84, 84), Image.ANTIALIAS)
    return image

def __save_image(image, iter, extension='.png'):
    # every 1/10 gets put in test set
    if np.mod(iter, 10) == 0:
        image.save('./record/test/' + str(iter) + extension)
    # rest go into train set
    else:
        image.save('./record/train/' + str(iter) + extension)
    return None


'''
Main function
'''
if __name__ == '__main__':
    make_dataset(extension='.png')
