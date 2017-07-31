#include "libfastree.h"

#include <math.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

void c_incompleteomptree(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads){
  int i;

  // Set root cell
  cell C0[N];
  cell *CN = C0;
  double p;
  initialize(C0,NCRIT);

  C0->xc = C0->yc = C0->zc = C0->r = 0.5;
  // Build tree
  for( i=0; i<N; i++ ) {
    cell *C = C0;
    while( C->nleaf >= NCRIT ) {
      C->nleaf++;
      int octant = (x[i] > C->xc) + ((y[i] > C->yc) << 1) + ((z[i] > C->zc) << 2);
      if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN,NCRIT);
      C = C->child[octant];
    }
    C->leaf[C->nleaf++] = i;
    if( C->nleaf >= NCRIT ) split_cell(x,y,z,C,CN,NCRIT);
  }

  // Multipole expansion
  getMultipole(C0,x,y,z,m,NCRIT);
  // Upward translation
  for( cell *C=CN; C!=C0; --C ) {
    cell *P = C->parent;
    upwardSweep(C,P);
  }
  #pragma omp parallel num_threads(threads) shared(C0,CN, potential) private(p,i)
  {
    // Evaluate expansion
    #pragma omp for
    for(i=0; i<N; i++){
        cell *C = C0;
        p = -m[i] / sqrtf(EPS2);
        evaluate(C,x,y,z,m,p,i,NCRIT,THETA,EPS2);
        potential[i] = p;
    }
  }

  return;
}
