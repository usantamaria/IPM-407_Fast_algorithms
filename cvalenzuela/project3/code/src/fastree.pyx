import cython

import numpy as np
cimport numpy as np

"""
  Import C/C++ functions from source
"""
cdef  extern from "libfastree.h":
  void c_direct_sum(int N, double* potential,double* x,double* y,double* z,double* m, double eps2)

  void c_treecode(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2)

  void c_threadcode(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2, int threads)

  void c_omptree(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2, int threads)

  void c_incompleteomptree(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2, int threads)

  void c_omptreemultipole(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2, int threads)


"""
  Direct Summation
"""
def direct_sum(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2):
  cdef np.int32_t n = particles.shape[0]
  cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
  cdef np.ndarray[double,ndim=1] x = particles[:,0]
  cdef np.ndarray[double,ndim=1] y = particles[:,1]
  cdef np.ndarray[double,ndim=1] z = particles[:,2]

  c_direct_sum(<int> n, <double*> &potential[0], <double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2);

  return potential

"""
  Calculate secuential Treecode
"""
def treecode(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2, theta, max_particles):
  cdef np.int32_t n = particles.shape[0]
  cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
  cdef np.ndarray[double,ndim=1] x = particles[:,0]
  cdef np.ndarray[double,ndim=1] y = particles[:,1]
  cdef np.ndarray[double,ndim=1] z = particles[:,2]

  c_treecode(<double> theta,<int> max_particles,<int> n, <double*> &potential[0],<double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2)

  return potential

"""
  Calculate Treecode Using Pthread (M2P Only)
"""
def threadcode(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2, theta, max_particles, threads):
  cdef np.int32_t n = particles.shape[0]
  cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
  cdef np.ndarray[double,ndim=1] x = particles[:,0]
  cdef np.ndarray[double,ndim=1] y = particles[:,1]
  cdef np.ndarray[double,ndim=1] z = particles[:,2]

  c_threadcode(<double> theta,<int> max_particles,<int> n, <double*> &potential[0],<double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2, <int> threads)

  return potential

"""
  Calculate "Full"-Treecode using OpenMP (M2P-M2M-M2P)
"""
def omptree(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2, theta, max_particles, threads):
    cdef np.int32_t n = particles.shape[0]
    cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
    cdef np.ndarray[double,ndim=1] x = particles[:,0]
    cdef np.ndarray[double,ndim=1] y = particles[:,1]
    cdef np.ndarray[double,ndim=1] z = particles[:,2]

    c_omptree(<double> theta,<int> max_particles,<int> n, <double*> &potential[0],<double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2, <int> threads)

    return potential

"""
  Calculate Evaluation-Treecode using OpenMP (M2P)
"""
def omptree_evaluation(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2, theta, max_particles, threads):
    cdef np.int32_t n = particles.shape[0]
    cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
    cdef np.ndarray[double,ndim=1] x = particles[:,0]
    cdef np.ndarray[double,ndim=1] y = particles[:,1]
    cdef np.ndarray[double,ndim=1] z = particles[:,2]

    c_incompleteomptree(<double> theta,<int> max_particles,<int> n, <double*> &potential[0],<double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2, <int> threads)

    return potential

"""
  Calculate Multipole-Treecode using OpenMP (M2P)
"""
def omptree_multipole(np.ndarray[double,ndim=2,mode="c"] particles, np.ndarray[double,ndim=1,mode="c"] mass, eps2, theta, max_particles, threads):
    cdef np.int32_t n = particles.shape[0]
    cdef np.ndarray[double,ndim=1] potential = np.zeros(n,dtype=np.float64)
    cdef np.ndarray[double,ndim=1] x = particles[:,0]
    cdef np.ndarray[double,ndim=1] y = particles[:,1]
    cdef np.ndarray[double,ndim=1] z = particles[:,2]

    c_omptreemultipole(<double> theta,<int> max_particles,<int> n, <double*> &potential[0],<double*> &x[0], <double*> &y[0], <double*> &z[0], <double*> &mass[0],<double> eps2, <int> threads)

    return potential
