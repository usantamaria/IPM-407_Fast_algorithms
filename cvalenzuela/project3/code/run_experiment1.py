import numpy as np
import fastree
import time
import os
from subprocess import call

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

np.random.seed(0)
eps=1e-10

theta = 0.5
max_n = 10

treecode_times = []
pthread_times = []
omp_times = []
omp_evaluation_times = []
omp_multipole_times = []


pthread_speedup = []
omp_speedup = []
omp_evaluation_speedup = []
omp_multipole_speedup = []


threads = 24
N_list = np.arange(100,1000001,5000)
for N in N_list:
        p = np.random.rand(N,3)
        m = np.random.rand(N)

        treecode_t = time.time()
        treecode = fastree.treecode(p,m,eps,theta,max_n)
        treecode_t = time.time() - treecode_t
        treecode_times.append(treecode_t)

        ptreecode_t = time.time()
        ptreecode = fastree.threadcode(p,m,eps,theta,max_n,threads)
        ptreecode_t = time.time() - ptreecode_t
        print("Pthread Treecode Time:", round(ptreecode_t,4), " Speedup:",round(treecode_t/ptreecode_t,2))

        if np.allclose(treecode,ptreecode):
                pthread_times.append(ptreecode_t)
                pthread_speedup.append(treecode_t/ptreecode_t)


        omptree_t = time.time()
        omptree = fastree.omptree(p,m,eps,theta,max_n, threads)
        omptree_t = time.time() - omptree_t
        print('"Full" OpenMP Treecode Time:', round(omptree_t,4), " Speedup:",round(treecode_t/omptree_t,2))

        if np.allclose(treecode, omptree):
                omp_times.append(omptree_t)
                omp_speedup.append(treecode_t/omptree_t)

        omptree2_t = time.time()
        omptree2 = fastree.omptree_evaluation(p,m,eps,theta,max_n,threads)
        omptree2_t = time.time() - omptree2_t
        print("OpenMP - Evaluation Treecode Time:", round(omptree2_t,4), " Speedup:",round(treecode_t/omptree2_t,2))
        if np.allclose(treecode,omptree2):
            omp_evaluation_times.append(omptree2_t)
            omp_evaluation_speedup.append(treecode_t/omptree2_t)


        omptree3_t = time.time()
        omptree3 = fastree.omptree_multipole(p,m,eps,theta,max_n,threads)
        omptree3_t = time.time() - omptree3_t
        print("OpenMP - Multipole Time:", round(omptree3_t,4), " Speedup:",round(treecode_t/omptree3_t,2))

        if np.allclose(treecode,omptree3):
            omp_multipole_times.append(omptree3_t)
            omp_multipole_speedup.append(treecode_t/omptree3_t)

pthread_times = np.array(pthread_times)
pthread_speedup = np.array(pthread_speedup)
omp_times = np.array(omp_times)
omp_speedup = np.array(omp_speedup)

np.savez("particles",N_list = N_list,treecode_times = treecode_times,pthread_times=pthread_times,
                 omp_times=omp_times, pthread_speedup=pthread_speedup,
                 omp_speedup=omp_speedup, omp_evaluation_times=omp_evaluation_times,
                  omp_evaluation_speedup = omp_evaluation_speedup, omp_multipole_times=omp_multipole_times,
                  omp_multipole_speedup=omp_multipole_speedup)
