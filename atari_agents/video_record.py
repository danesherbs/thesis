'''
Script originally written by Ben Goodrich from ALE as a direct
port of example provided in doc/examples/sharedLibraryInterfaceExample.cpp
'''

import sys
from random import randrange
from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt
import numpy as np


if len(sys.argv) < 2:
    print('Usage: %s rom_file' % sys.argv[0])
    sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
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

# Load the ROM file
rom_file = str.encode(sys.argv[1])
ale.loadROM(rom_file)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()

# make recording directories
import os
if not os.path.exists('record'):
    os.makedirs('record')

# initialise iteration counter
iter = 0

# play game
for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        # capture screenshot
        screenshot = ale.getScreenRGB()
        # every 1/10 gets put in test set
        plt.imsave('./record/' + str(iter), screenshot)
        a = legal_actions[randrange(len(legal_actions))]
        # apply an action and get the resulting reward
        reward = ale.act(a);
        total_reward += reward
        # increment iteration counter
        iter += 1
    print('Episode %d ended with score: %d' % (episode, total_reward))
    ale.reset_game()
