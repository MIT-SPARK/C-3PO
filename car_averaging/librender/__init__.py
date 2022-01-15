import ctypes
import os, sys

pyrender_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, pyrender_dir)
ctypes.cdll.LoadLibrary(os.path.join(pyrender_dir, '../pyrender.so'))
from pyrender import *
