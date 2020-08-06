import os
import sys
dir = os.path.split(os.path.realpath(__file__))[0]
dir = os.path.join(dir, '..')
sys.path.append(dir)