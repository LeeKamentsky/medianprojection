'''setup.py - handle distutils operations for the medianprojection package

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

from numpy import get_include
import os
import sys

is_win = sys.platform == "win32"

def configuration():
    if is_win:
        extra_compile_args = None
        extra_link_args = ['/MANIFEST']
    else:
        extra_compile_args = ['-O3']
        extra_link_args = None
    pyx_path = os.path.join("medianprojection", "_3dmedianfilter.pyx")
    extensions = [Extension(
        name="_3dmedianfilter",
        sources=[pyx_path],
        include_dirs=[get_include()],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args)]
    
    return { "name":"medianprojection",
             "version":"1.0.0",
             "packages":["medianprojection"],
             "description":"An application that calculates the median projection of an image stack",
             "maintainer":"Lee Kamentsky",
             "maintainer_email":"leek@broadinstitute.org",
             "cmdclass":{'build_ext': build_ext},
             "ext_modules": cythonize(extensions),
             "classifiers":['Development Status :: 5 - Production/Stable',
                            'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                          ],
             "install_requires":['javabridge', 'python-bioformats'],
             "entry_points":{
                 'console_scripts':['medianprojection=medianprojection.main:main']
                 }
             }

if __name__=='__main__':
    setup(**configuration())