import os
import random

filenames = os.listdir('images')
random.shuffle(filenames)
filenames = filenames[10000:]

for file in filenames:
    os.remove('images/'+file)
