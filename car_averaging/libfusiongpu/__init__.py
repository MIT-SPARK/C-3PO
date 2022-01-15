import ctypes
import os, sys

pyfusion_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, pyfusion_dir)
ctypes.cdll.LoadLibrary(os.path.join(pyfusion_dir, 'build', 'libfusion_gpu.so'))
from cyfusion import *
