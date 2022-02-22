import os
import platform

if platform.system() == 'Windows':
    os.system('cls')
else:
    os.system('clear')
os.system('tensorboard --logdir=runs --bind_all --port=8000')
