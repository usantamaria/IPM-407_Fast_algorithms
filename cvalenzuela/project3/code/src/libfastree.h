#ifndef FASTREE_LIB_H
#define FASTREE_LIB_H


struct cell{
  int nleaf,nchild;
  int* leaf;
  double xc,yc,zc,r;
  double multipole[10];
  cell *parent,*child[8];
};

// Calculate the potential using direct summation
void c_direct_sum(int N, double* potential,double* x,double* y,double* z,double* m, double eps2);

// Treecode implementation
void c_treecode(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2);

// PThread implementation
void c_threadcode(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads);

//Full OpenMp implementation
void c_omptree(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads);
//OpenMP only evaluation
void c_incompleteomptree(double THETA,int N_CRIT,int N, double* potential,double* x,double* y,double* z,double* m, double eps2, int threads);
//OpenMP Evaluation and P2M
void c_omptreemultipole(double THETA,int NCRIT,int N, double* potential,double* x,double* y,double* z,double* m, double EPS2, int threads);

void initialize(cell *C, int NCRIT);
void add_child(int octant, cell *C, cell *&CN, int NCRIT);
void split_cell(double *x, double *y, double *z, cell *C, cell *&CN,int NCRIT);
void getMultipole(cell *C, double *x, double *y, double *z, double *m, int NCRIT);
void upwardSweep(cell *C, cell *P);
void evaluate(cell *C, double *x, double *y, double *z, double *m, double &p, int i, int NCRIT, double THETA, double EPS2);

#endif
