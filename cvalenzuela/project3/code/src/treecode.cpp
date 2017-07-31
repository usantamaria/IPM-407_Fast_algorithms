#include "libfastree.h"
#include <stdlib.h>
#include <math.h>

void initialize(cell *C, int NCRIT) {
  C->nleaf = C->nchild = 0;
  C->parent = NULL;
  for( int i=0; i<8; i++ ) C->child[i] = NULL;
  for( int i=0; i<10; i++ ) C->multipole[i] = 0;
  C->leaf = (int*) malloc(sizeof(int)*NCRIT);
}

void add_child(int octant, cell *C, cell *&CN, int NCRIT) {
  ++CN;
  initialize(CN, NCRIT);
  CN->r  = C->r/2;
  CN->xc = C->xc+CN->r*((octant&1)*2-1);
  CN->yc = C->yc+CN->r*((octant&2)-1);
  CN->zc = C->zc+CN->r*((octant&4)/2-1);
  CN->parent = C;
  C->child[octant] = CN;
  C->nchild |= (1 << octant);
}

void split_cell(double *x, double *y, double *z, cell *C, cell *&CN,int NCRIT) {
  for( int i=0; i<NCRIT; i++ ) {
    int l = C->leaf[i];
    int octant = (x[l] > C->xc) + ((y[l] > C->yc) << 1) + ((z[l] > C->zc) << 2);
    if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN, NCRIT);
    cell *CC = C->child[octant];
    CC->leaf[CC->nleaf++] = l;
    if( CC->nleaf >= NCRIT ) split_cell(x,y,z,CC,CN, NCRIT);
  }
}

void getMultipole(cell *C, double *x, double *y, double *z, double *m, int NCRIT) {
  double dx,dy,dz;
  if( C->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ )
      if( C->nchild & (1 << c) ) getMultipole(C->child[c],x,y,z,m,NCRIT);
  } else {
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
}

void upwardSweep(cell *C, cell *P) {
  double dx,dy,dz;
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

void evaluate(cell *C, double *x, double *y, double *z, double *m, double &p, int i, int NCRIT, double THETA, double EPS2) {
  double dx,dy,dz,r,X,Y,Z,R,R3,R5;
  if( C->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ )
      if( C->nchild & (1 << c) ) {
        cell *CC = C->child[c];
        dx = x[i]-CC->xc;
        dy = y[i]-CC->yc;
        dz = z[i]-CC->zc;
        r = sqrtf(dx*dx+dy*dy+dz*dz);
        if( CC->r > THETA*r ) {
          evaluate(CC,x,y,z,m,p,i, NCRIT, THETA, EPS2);
        } else {
          X = x[i]-CC->xc;
          Y = y[i]-CC->yc;
          Z = z[i]-CC->zc;
          R = sqrtf(X*X+Y*Y+Z*Z);
          R3 = R*R*R;
          R5 = R3*R*R;
          p += CC->multipole[0]/R;
          p += CC->multipole[1]*(-X/R3);
          p += CC->multipole[2]*(-Y/R3);
          p += CC->multipole[3]*(-Z/R3);
          p += CC->multipole[4]*(3*X*X/R5-1/R3);
          p += CC->multipole[5]*(3*Y*Y/R5-1/R3);
          p += CC->multipole[6]*(3*Z*Z/R5-1/R3);
          p += CC->multipole[7]*(3*X*Y/R5);
          p += CC->multipole[8]*(3*Y*Z/R5);
          p += CC->multipole[9]*(3*Z*X/R5);
        }
      }
  } else {
    for( int l=0; l<C->nleaf; l++ ) {
      int j = C->leaf[l];
      dx = x[i]-x[j];
      dy = y[i]-y[j];
      dz = z[i]-z[j];
      r = sqrtf(dx*dx+dy*dy+dz*dz+EPS2);
      p += m[j] / r;
    }
  }
}

void c_treecode(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2){
  // Set root cell
  cell C0[N];
  initialize(C0,NCRIT);
  C0->xc = C0->yc = C0->zc = C0->r = 0.5;
  // Build tree
  cell *CN = C0;
  for( int i=0; i<N; i++ ) {
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
  // Evaluate expansion
  for( int i=0; i<N; i++ ) {
    cell *C = C0;
    double p = -m[i] / sqrtf(EPS2);
    evaluate(C,x,y,z,m,p,i,NCRIT,THETA,EPS2);
    potential[i] = p;
  }

    return;
}
