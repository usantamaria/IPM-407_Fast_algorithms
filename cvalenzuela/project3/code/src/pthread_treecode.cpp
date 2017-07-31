#include "libfastree.h"
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#include <stdio.h>

double *xx,*yy,*zz,*mm,*ppotential;
int pNCRIT, pThreads, NN;
double pEPS2;
double pTHETA;
cell *pC;

struct thread_args{
  int thread_id;
  int N;
};

void *p_evaluate(void *arg){
  thread_args args = *((thread_args*) arg);

  int i = args.thread_id;
  int threads = pThreads;
  int N = NN;

  int start, end;
  int step = N/threads;
  start = i*step;
  if(i == threads-1){
    end = N;
  }else{
    end = (i+1)*step;
  }
  for(i=start; i< end; i++ ) {
    cell *C = pC;
    double p = -mm[i] / sqrtf(pEPS2);
    evaluate(C,xx,yy,zz,mm,p,i,pNCRIT,pTHETA,pEPS2);
    ppotential[i] = p;
  }
}

void *p_upwardSweep(void *arg){
  thread_args args = *((thread_args*) arg);

  int i = args.thread_id;
  int N = args.N;
  int gap = pThreads + i;
  int level = 1;
  int max_n = 0;
  int min_level = 0;
  int max_level = pNCRIT * level;
  while(max_n < NN/8){
    printf("########################\n LEVEL %d \n ########################\n", level);
    for(int j = N-i; j > 0; j=j-gap){
      cell* C = &(pC[j]);
      if(C->nleaf < max_level && C->nleaf > min_level){
        cell* P = C -> parent;
        upwardSweep(C,P);
        if(C-> nleaf > max_n){
          max_n = C-> nleaf;
        }
        printf("j: %d nleaf: %d\n",j,C->nleaf);
      }
    }
    level++;
    min_level = (level-1) * pNCRIT;
    max_level = pNCRIT * level;
  }
}

void c_threadcode(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads){
  pthread_t *td;
  thread_args *args;
  int i;

  td = (pthread_t*) malloc(sizeof(pthread_t)* threads);
  args = (thread_args *) malloc(sizeof(args)*threads);

  xx = x;
  yy = y;
  zz = z;
  mm = m;
  ppotential = potential;
  pNCRIT = NCRIT;
  pTHETA = THETA;
  pEPS2 = EPS2;
  pThreads = threads;
  NN = N;

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

  //Setting Tree Root to Global
  pC = C0;
  // Multipole expansion
  getMultipole(C0,x,y,z,m,NCRIT);

  // Upward translation
  // int dis = (int) ((cell*) CN - (cell*)C0);
  //
  // for(i = 0; i < threads; i++){
  //   args[i].thread_id = i;
  //   args[i].N = dis;
  //   pthread_create(&td[i], NULL, p_upwardSweep,(void *)&args[i]);
  // }
  // for (i = 0; i < threads; i++) {
  //   pthread_join(td[i], NULL);
  // }

  //DELETE
  for( cell *C=CN; C!=C0; --C ) {
    cell *P = C->parent;
    upwardSweep(C,P);
  }

  // Evaluate expansion
  for(i = 0; i < threads; i++){
    args[i].thread_id = i;
    pthread_create(&td[i], NULL, p_evaluate,(void *)&args[i]);
  }
  for (i = 0; i < threads; i++) {
    pthread_join(td[i], NULL);
  }

}
