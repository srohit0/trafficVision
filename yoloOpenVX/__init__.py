# Copyright (c) 2018. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys, os
import ctypes
import numpy.ctypeslib as ctl

root_dir  = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
weights   = os.path.join(root_dir , 'models', 'openVXModel', 'weights.bin');
libdir    = os.path.join(root_dir , 'lib');
graph     = ctl.load_library('libannmodule', libdir)

interface                              = ctl.load_library('libannpython', libdir);
interface.annQueryInference.restype    = ctypes.c_char_p
interface.annQueryInference.argtypes   = []
interface.annCreateInference.restype   = ctypes.c_void_p
interface.annCreateInference.argtypes  = [ctypes.c_char_p]
interface.annReleaseInference.restype  = ctypes.c_int
interface.annReleaseInference.argtypes = [ctypes.c_void_p]



model                      = ctl.load_library('libyoloOpenVX',    libdir)
model.detectBoxes.restype  = ctypes.py_object
model.detectBoxes.argtypes = [ctypes.c_void_p, ctl.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")];

print ("Loaded", os.path.basename(os.path.dirname(__file__)))
