from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import os
import numpy as np
"""
    Creating Module using Cython
"""
setup(
    name="Treecode MPI Wrapper",
    ext_modules = cythonize( #Compile the .pyx to .pyc
        Extension(  #Create the module
            "fastree", #Module Name
            #List of sources needed by the module (.pyx)
            sources=["src/fastree.pyx","src/direct_sum.cpp",
                    "src/treecode.cpp","src/pthread_treecode.cpp",
                    "src/omptree.cpp","src/omptreeincomplete.cpp",
                    "src/ompmultipole.cpp"],
            #Compilter to use
            language="c++",
            #Including numpy dirs
            include_dirs=[np.get_include()],
            #Adding openmp flag to compiler
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
    )
)
