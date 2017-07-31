#include "libfastree.h"

#include <math.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>

void ompPrecalculate(cell *C, double *x, double *y, double *z, double *m, int NCRIT){
  double dx,dy,dz;
  if( C->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ ){
        if( C->nchild & (1 << c) ){
          #pragma omp task untied
          {
          ompPrecalculate(C->child[c],x,y,z,m,NCRIT);
          }
        }
    }
  }
  else
  {

    for( int l=0; l<C->nleaf; l++ ) {
      int j = C->leaf[l];
      dx = C->xc-x[j];
      dy = C->yc-y[j];
      dz = C->zc-z[j];
      C->multipole[0] += m[j];
      C->multipole[1] += m[j]*dx;
      C->multipole[2] += m[j]*dy;
      C->multipole[3] += m[j]*dz;
      C->multipole[4] += m[j]*dx*dx/2;
      C->multipole[5] += m[j]*dy*dy/2;
      C->multipole[6] += m[j]*dz*dz/2;
      C->multipole[7] += m[j]*dx*dy/2;
      C->multipole[8] += m[j]*dy*dz/2;
      C->multipole[9] += m[j]*dz*dx/2;
    }
  }
  #pragma omp taskwait
  #pragma omp critical
  {
    cell *P = C->parent;
    //upwardSweep
    if(P != NULL){
      dx = P->xc-C->xc;
      dy = P->yc-C->yc;
      dz = P->zc-C->zc;
      P->multipole[0] += C->multipole[0];
      P->multipole[1] += C->multipole[1]+ dx*C->multipole[0];
      P->multipole[2] += C->multipole[2]+ dy*C->multipole[0];
      P->multipole[3] += C->multipole[3]+ dz*C->multipole[0];
      P->multipole[4] += C->multipole[4]+ dx*C->multipole[1]+dx*dx*C->multipole[0]/2;
      P->multipole[5] += C->multipole[5]+ dy*C->multipole[2]+dy*dy*C->multipole[0]/2;
      P->multipole[6] += C->multipole[6]+ dz*C->multipole[3]+dz*dz*C->multipole[0]/2;
      P->multipole[7] += C->multipole[7]+(dx*C->multipole[2]+   dy*C->multipole[1]+dx*dy*C->multipole[0])/2;
      P->multipole[8] += C->multipole[8]+(dy*C->multipole[3]+   dz*C->multipole[2]+dy*dz*C->multipole[0])/2;
      P->multipole[9] += C->multipole[9]+(dz*C->multipole[1]+   dx*C->multipole[3]+dz*dx*C->multipole[0])/2;
    }
  }
}


void c_omptree(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads){
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

  #pragma omp parallel num_threads(threads) shared(C0,CN, potential) private(p,i)
  {
    // Multipole expansion
    #pragma omp single
    {
      ompPrecalculate(C0,x,y,z,m,NCRIT);
    }
    #pragma taskwait

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
