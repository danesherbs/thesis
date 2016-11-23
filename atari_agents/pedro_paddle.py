#!/usr/bin/env python
# python_example.py
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import random
from ale_python_interface import ALEInterface
import numpy as np

if len(sys.argv) < 2:
  print('Usage: %s rom_file' % sys.argv[0])
  sys.exit()

ale = ALEInterface()

# Get & Set the desired settings
ale.setInt(b'random_seed', random.randint(0, 999))

# Show screen and turn on sound
ale.setBool("display_screen", True)
ale.setBool("sound", True)

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

# Probability p of changing action
p = 0.15

# Choose initial action
a = random.choice(range(len(legal_actions)))

# Play 10 episodes
for episode in range(10):
  total_reward = ale.act(a)
  while not ale.game_over():
    if np.random.binomial(1, p):  # change action with probability p
        a = random.choice(range(len(legal_actions)))
    # Apply an action and get the resulting reward
    reward = ale.act(a);
    total_reward += reward
  print('Episode %d ended with score: %d' % (episode, total_reward))
  ale.reset_game()
