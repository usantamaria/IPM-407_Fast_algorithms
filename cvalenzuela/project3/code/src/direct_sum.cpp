#include "libfastree.h"
#include <math.h>
#include <stdlib.h>

void c_direct_sum(int N, double* potential,double* x,double* y,double* z,double* m, double eps2){
  float dx,dy,dz,r;

  for( int i=0; i<N; i++ ) {
    float p = -m[i]/sqrtf(eps2);
    for( int j=0; j<N; j++ ) {
      dx = x[i]-x[j];
      dy = y[i]-y[j];
      dz = z[i]-z[j];
      r = sqrtf(dx*dx+dy*dy+dz*dz+eps2);
      p += m[j] / r;
    }
    potential[i] = p;
  }
  return;
}
