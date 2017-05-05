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


def make_dataset(num_games, pre_processing=True, split_into_train_test=True):
    if len(sys.argv) < 2:
        print('Usage: %s rom_file' % sys.argv[0])
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

    # get the list of legal actions
    legal_actions = ale.getLegalActionSet()

    # make recording directory
    import os
    if not os.path.exists('./record/'):
        os.makedirs('./record/')
    if split_into_train_test:
        if not os.path.exists('./record/train'):
            os.makedirs('./record/train')
        if not os.path.exists('record/test'):
            os.makedirs('./record/test')

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
                # pre-process image if needed
                if pre_processing:
                    screenshot = __pre_process(screenshot)
                # save screenshot in appropriate directory
                __save_screenshot(screenshot, split_into_train_test, iter/2)
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

def __pre_process(image_array):
    '''
    Takes numpy array and returns processed numpy array.
    Processing extracts luminance and rescales to 84x84.
    '''
    image = Image.fromarray(image_array)
    image = image.convert('L')
    image = image.resize((84, 84), Image.ANTIALIAS)
    image_array = np.asarray(image)
    return image_array

def __save_screenshot(screenshot, split_into_train_test, iter):
    # save screenshot in appropriate directory
    if split_into_train_test:
        # every 1/10 gets put in test set
        if np.mod(iter, 10) == 0:
            plt.imsave('./record/test/' + str(iter), screenshot, cmap='gray')
        # rest go into train set
        else:
            plt.imsave('./record/train/' + str(iter), screenshot, cmap='gray')
    else:
        # everything goes in record directory
        plt.imsave('./record/' + str(iter), screenshot, cmap='gray')
    return None

if __name__ == '__main__':
    num_games = 1
    pre_processing = True
    split_into_train_test = True
    make_dataset(1, pre_processing=pre_processing, split_into_train_test=split_into_train_test)