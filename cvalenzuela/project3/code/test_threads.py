import numpy as np
import fastree
import time


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
N= 40000
eps=1e-10

theta = 0.5
max_n = 10
p = np.random.rand(N,3)
m = np.random.rand(N)

treecode_t = time.time()
treecode = fastree.treecode(p,m,eps,theta,max_n)
treecode_t = time.time() - treecode_t

threads_max = 4
for threads in range(1,threads_max+1):
    print(bcolors.HEADER+"################ Threads:{}################".format(threads)+ bcolors.ENDC)
    ptreecode_t = time.time()
    ptreecode = fastree.threadcode(p,m,eps,theta,max_n,threads)
    ptreecode_t = time.time() - ptreecode_t
    print("Treecode Time:",round(treecode_t,4), " Speedup:",round(treecode_t/treecode_t,2))
    print("Pthread Treecode Time:", round(ptreecode_t,4), " Speedup:",round(treecode_t/ptreecode_t,2))
    if np.allclose(treecode,ptreecode):
        msg = bcolors.OKGREEN+str(np.allclose(treecode,ptreecode))+ bcolors.ENDC
    else:
        msg = bcolors.FAIL+str(np.allclose(treecode,ptreecode))+ bcolors.ENDC
    print("Equal?",msg)
    omptree_t = time.time()
    omptree = fastree.omptree(p,m,eps,theta,max_n,threads)
    omptree_t = time.time() - omptree_t
    print("OpenMP Treecode Time:", round(omptree_t,4), " Speedup:",round(treecode_t/omptree_t,2))
    if np.allclose(treecode,omptree):
        msg = bcolors.OKGREEN+str(np.allclose(treecode,omptree))+ bcolors.ENDC
    else:
        msg = bcolors.FAIL+str(np.allclose(treecode,omptree))+ bcolors.ENDC
    print("Equal?",msg)

    omptree2_t = time.time()
    omptree2 = fastree.omptree_evaluation(p,m,eps,theta,max_n,threads)
    omptree2_t = time.time() - omptree2_t
    print("OpenMP - Evaluation Treecode Time:", round(omptree2_t,4), " Speedup:",round(treecode_t/omptree2_t,2))
    if np.allclose(treecode,omptree2):
        msg = bcolors.OKGREEN+str(np.allclose(treecode,omptree2))+ bcolors.ENDC
    else:
        msg = bcolors.FAIL+str(np.allclose(treecode,omptree2))+ bcolors.ENDC
    print("Equal?",msg)

    omptree3_t = time.time()
    omptree3 = fastree.omptree_multipole(p,m,eps,theta,max_n,threads)
    omptree3_t = time.time() - omptree3_t
    print("OpenMP - Multipole Time:", round(omptree3_t,4), " Speedup:",round(treecode_t/omptree3_t,2))
    if np.allclose(treecode,omptree3):
        msg = bcolors.OKGREEN+str(np.allclose(treecode,omptree3))+ bcolors.ENDC
    else:
        msg = bcolors.FAIL+str(np.allclose(treecode,omptree3))+ bcolors.ENDC
    print("Equal?",msg)
