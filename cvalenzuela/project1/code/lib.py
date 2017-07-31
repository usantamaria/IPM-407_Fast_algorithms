import numpy as np
import time
import matplotlib.pyplot as plt

from numba import jit,int32,double,boolean
np.seterr(all='raise')

"""
Create the exact solution for the poisson equation

Parameters
-----------------
m : int
    Number of nodes of the grid for each dimension.

Returns
------------------
u : 2D Array
    Exact Solution for that number of nodes.

"""
@jit(double[:,:](int32),nopython=True,nogil=True)
def create_solution(m):
    x = np.linspace(0,1,m+2)[1:-1]
    y = np.linspace(0,1,m+2)[1:-1]
    
    
    n_x = len(x)
    n_y = len(y)
    u= np.zeros((n_x,n_y))
    for i,x_i in enumerate(x): 
        for j,y_j in enumerate(y):
            u[i,j] = (x_i**2-x_i**4)*(y_j**4-y_j**2)
    return u


"""
Gauss seidel iteration implementation for Poisson Equation.
Parameters
-------------------
x : 2D Numpy Array
    Current solution.

b : 2D Numpy Array
    Right side of the Poisson Equation.

Returns
-------------------
x : 2D Numpy Array
    Solution after relaxation.
"""
@jit(double[:,:](double[:,:],double[:,:], double),nopython=True,nogil=True)
def gauss_seidel_iteration(x, b, h = None):

    m = x.shape[0]
    
    next_iter = np.zeros_like(x)

    if not h:
        h = 1./(m-1)
    
    for i in range(m):
        for j in range(m):
            next_iter[i,j] = b[i,j] * h**2 
            
            #Check if we aren't in a boundary
            #otherwise we substract 0
            if i > 0:
                next_iter[i,j] -= next_iter[i-1,j]
            if i < (m -1):
                next_iter[i,j] -= x[i+1,j]
            if j > 0:
                next_iter[i,j] -= next_iter[i,j-1]
            if j < (m -1):
                next_iter[i,j] -= x[i,j+1]
            
            next_iter[i,j] /= -4.
    return next_iter

"""
Create the right part of the Poisson Equation using the grid data.
Parameters
---------------
x : 1D Array
    x values of the grid (Horizontal Position of the node)

y: 1D Array
    y values of the grid (Vertical Position of the node)

Returns
--------------
b : 2D Array
    Right side of the poisson equation.
"""
@jit(double[:,:](double[:],double[:]),nopython=True,nogil=True)
def create_right(x,y):
    m = len(x)
    b = np.zeros((m,m))
    
    for i in range(m):
        for j in range(m):
            b[i,j] = - 2 * ((1-6*x[i]**2)  
                             *y[j]**2 
                            *(1-y[j]**2) 
                            +(1-6*y[j]**2)
                            *x[i]**2 
                            *(1 - x[i]**2))
    return b


"""
Calculate the residue for the current solution.
Parameters
----------------
x : 2D Numpy Array
    Current solution.
b : 2D Numpy Array
    Right side of the equation.

Returns
-----------------
errors : 2D Numpy Aray
        Residue for the current solution.
"""
@jit(double[:,:](double[:,:],double[:,:], double),nopython=True,nogil=True)
def calculate_residue(x,b, h = None):
    n_x,n_y = x.shape
    A_x = np.zeros((n_x,n_y))
    for i in range(n_x):
        for j in range(n_y):
            node_sum = 0
            if i > 0:
                node_sum += x[i-1,j]/h**2

            if j > 0:
                node_sum += x[i,j-1]/h**2

            node_sum += -4*x[i,j]/h**2
            
            if i+1 != n_x :
                node_sum += x[i+1,j]/h**2
            
            if j+1 != n_y:
                node_sum += x[i,j+1]/h**2
            A_x[i,j] =  node_sum
    errors = b - A_x
    return errors


"""
Gauss seidel Implementation. 
Parameters
-------------------------------
m : int
    Number of nodes
atol: float
    Minimun tolerance before stop
plot: boolean
    Plot the current approximantion in each step (A lot slower)

random_seed: int
    Value to generate a random initial guess (-1 used for initial guess = 0)

Returns
------------------------------
v: 2D Numpy Array
    The solution after relaxation.

iters: int
    Number of iterations.
"""
def gauss_seidel(m, atol = 1e-3, plot=False, random_seed = -1, demo = False):
        
    if plot:
        fig,ax = plt.subplots(1,1)
        
    x = np.linspace(0,1,m+2)[1:-1]
    y = np.linspace(0,1,m+2)[1:-1]
    
    h = x[1] - x[0]
    
    if random_seed > 0:
        np.random.seed(random_seed)
        v = np.random.rand(m,m)
    else:
        v = np.zeros((m,m))
        
    f = create_right(x,y)
    
    r = np.linalg.norm(calculate_residue(v,f,h))
    iters = 0
    
    while r > atol:
        v = gauss_seidel_iteration(v,f,h)
        
        r = np.linalg.norm(calculate_residue(v,f,h))
        
        iters += 1

        if plot:
            ax.set_title("Iteration: {}, Residue: {}".format(iters,r))
            ax.imshow(v)
            fig.canvas.draw()
        if demo and iters == 20:
            break
    return v,iters

"""
Restrict the matrix to a coarser grid using injection or fullweighted
Parameters
------------------------
m : int
    Number of nodes
fullweight: boolean
    If true uses fullweighted restriction, otherwise uses injection.

Returns
------------------------
rC : 2D Array
    The matrix in the coarser grid.
"""
@jit(double[:,:](double[:,:], boolean), nopython=True,nogil=True)
def restrict(r, fullweight=True):
    m = len(r)
    if m %2==0:
        mC = int(m/2)
    else:
        mC = int((m-1)/2+1)

    rC = np.zeros((mC,mC))

    for i in range(mC):
        for j in range(mC):
            rC[i,j] = r[2*i,2*j]

            #Injection doesn't use this ones
            if fullweight:
                if 2*i > 0:
                    rC[i,j] += 0.5 * r[2*i-1,2*j]
                if 2*i < m -1:            
                    rC[i,j] += 0.5 * r[2*i+1,2*j]
                if 2*j > 0:
                    rC[i,j] += 0.5 * r[2*i,2*j-1]
                if 2*j < m -1:            
                    rC[i,j] += 0.5 * r[2*i,2*j+1]


                # if 2*i > 0 and 2*j < m-1:            
                #     rC[i,j] += 0.25 * r[2*i-1,2*j+1]
                # if 2*i < m-1 and 2*j < m-1:
                #     rC[i,j] += 0.25 * r[2*i+1,2*j+1]
                # if 2*i > 0 and 2*j > 0:                
                #     rC[i,j] += 0.25 * r[2*i-1,2*j-1]
                # if 2*i < m-1 and 2*j > 0:                    
                #     rC[i,j] += 0.25 * r[2*i+1,2*j-1]
                
            rC[i,j] *= 0.25    

    return rC

"""
Interpolate the matrix to a finer grid using 8 neightbors for each
point in the coarser grid.
Parameters
---------------------------------------
rC: 2D Array
    Coarse Grid residue

Returns
---------------------------------------
r : 2D Array
    The residue in the finer grid.
"""
@jit(double[:,:](double[:,:],int32), nopython=True,nogil=True)
def interpolate(rC, m = None):
    if not m:
        m = ((rC.shape[0]-1)*2)+1

    r = np.zeros((m,m))


    for i in range(rC.shape[0]):
        for j in range(rC.shape[1]):

            r[2*i, 2*j] = rC[i,j]

            if 2*i < m-1:
                r[2*i+1,2*j] = rC[i,j]
                if i < rC.shape[0]-1:
                    r[2*i+1,2*j] += rC[i+1,j]
                r[2*i+1,2*j] *= 0.5

            if 2*j < m-1:
                r[2*i,2*j+1] = rC[i,j]
                if j < rC.shape[1]-1:
                    r[2*i,2*j+1] += rC[i,j+1]
                r[2*i,2*j+1] *= 0.5

            if 2*i < m-1 and 2*j < m-1:
                r[2*i+1,2*j+1] = rC[i,j]
                if i < rC.shape[0]-1:
                    r[2*i+1,2*j+1] += rC[i+1,j]
                if j < rC.shape[1]-1:
                    r[2*i+1,2*j+1] += rC[i,j+1]
                if i < rC.shape[0]-1 and j < rC.shape[1]-1:
                    r[2*i+1,2*j+1] += rC[i+1,j+1]
                r[2*i+1,2*j+1] *= 0.25 

    # # 8 neightbors
    # for i in range(rC.shape[0]):
    #     for j in range(rC.shape[1]):
            
    #         r[2*i,2*j] +=  rC[i,j]
            
    #         if 2*i > 0:
    #             r[2*i -1,2*j] += 1/2. * rC[i,j]
    #         if 2*i < m-1:
    #             r[2*i +1,2*j] += 1/2. * rC[i,j]
    #         if 2*j > 0:
    #             r[2*i,2*j -1] += 1/2. * rC[i,j]
    #         if 2*j < m -1:
    #             r[2*i,2*j +1] += 1/2. * rC[i,j]
            
    #         if 2*i > 0 and 2*j < m-1:
    #             r[2*i-1,2*j +1] += 1/4. * rC[i,j]
    #         if 2*i < m-1 and 2*j < m-1:    
    #             r[2*i+1,2*j +1] += 1/4. * rC[i,j]
    #         if 2*i > 0 and 2*j > 0:
    #             r[2*i-1,2*j -1] += 1/4. * rC[i,j]
    #         if 2*i < m-1 and 2*j > 0:    
    #             r[2*i+1,2*j -1] += 1/4. * rC[i,j]
    return r


"""
Flatten the 2D array to 1D.
Parameters
--------------
x : int
    First index
y : int
    Second index
m : int
    #First indices.

Returns
--------------
index in the linearized matrix
"""
@jit(int32(int32,int32,int32), nopython=True, nogil=True)
def linearize(x,y,m):
    return y+x*m


"""
Create the full dense system for poisson equation.
Parameters
--------------------
m : int
    Number of nodes.

h : float
    Distance between nodes

Return
--------------------
A : 2D Array.
    Dense matrix for the flatten 2D Matrix. (The first row correspond to the 
    first element of the grid [0,0]).
"""
@jit(double[:,:](int32,double), nopython=True,nogil=True)
def create_system(m,h=None):
    if not h:
        h = 1./(m-1)
    
    n = m**2
    A = np.zeros((n,n))
    for i in range(m):
        for j in range(m):
            index = linearize(i,j,m)
            if i==0 or i == m-1 or j == 0 or j == m -1:
                if i!= 0:
                    A[index, linearize(i-1,j,m)] = 1./(h**2) 
                if i != m-1:
                    A[index, linearize(i+1,j,m)] = 1./(h**2) 
                if j != 0:
                    A[index, linearize(i,j-1,m)] = 1./(h**2)
                if j != m-1:
                    A[index, linearize(i,j+1,m)] = 1./(h**2)
                A[index, linearize(i,j,m)] = -4./(h**2)
            else:
                A[index, linearize(i-1,j,m)] = 1./(h**2)
                A[index, linearize(i+1,j,m)] = 1./(h**2)

                A[index, linearize(i,j,m)] = -4/(h**2)

                A[index, linearize(i,j-1,m)] = 1./(h**2)
                A[index, linearize(i,j+1,m)] = 1./(h**2)

            
    return A

@jit(double[:](double[:,:]), nopython=True,nogil=True)
def flatten(b):
    x,y = b.shape
    b_f = np.zeros(x*y)
    for i in range(x):
        for j in range(y):
            b_f[linearize(i,j,x)] = b[i,j]
    return b_f

@jit(double[:,:](double[:]), nopython=True,nogil=True)
def unflatten(x):
    m = int(np.sqrt(len(x)))
    b_uf = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            b_uf[i,j] = x[linearize(i,j,m)]
    return b_uf 

"""
A single iteration of V-Cycle using the recursive form.
Parameters
----------------------
x : 2D Numpy Array
    Current Solution
b : 2D Numpy Array
    Right side of the equation
v1: int
    Number of iterations before coarsening
v2: int
    Number of iterations after coarsening
fullweight: boolean
    If using fullweight or injection as restriction
plot, ax, fig: boolean, plt.axes, plt.figure
    Variables used for the interactive/visual demostration

Returns
----------------------
x: 2D Numpy array
    Solution after relaxation
ops: float
    Number of Gauss Seidel relaxation compared
    to the same number of nodes.
"""
def vcycle_iteration(x,b,min_m = None, h = None, v1 = 2, v2=2, fullweight=True,plot=False, ax=None, fig=None):
    if not min_m:
        min_m = 3
    m = x.shape[0]
    if not h:
        h = 1./(m-1)

    if m > min_m:
        #Do v1 gauss_seidel iterations
        for it in range(v1):
            x = gauss_seidel_iteration(x,b,h)
            
            if plot and fig and ax:
                r = np.linalg.norm(calculate_residue(x,b,h))

                ax.set_title("V1 ({}): Resolution: {}x{} \n Residue: {}".format(it,m,m,r))
                ax.imshow(x)
                fig.canvas.draw()  
                #time.sleep(1)

        r = calculate_residue(x,b,h)

        #Restrict r
        residC = restrict(r,fullweight=fullweight)
        
        #Calculate Ae=r in coarser grid
        mC = len(residC)
        hC = 1./(mC-1)
        eC,iter_ops = vcycle_iteration(np.zeros((mC,mC)),residC,min_m = min_m,h = hC, v1 = v1, v2=v2, fullweight=fullweight,plot=plot, ax=ax, fig=fig)

        iter_ops /= 4.
        
        #Interpolate x
        x = x + interpolate(eC,m=m)
        
        #Run v2 times gauss seidel iterations
        for it in range(v2):
            x = gauss_seidel_iteration(x,b,h)
            
            if plot and fig and ax:
                r = np.linalg.norm(calculate_residue(x,b,h))

                ax.set_title("V1 ({}): Resolution: {}x{} \n Residue: {}".format(it,m,m,r))
                ax.imshow(x)
                fig.canvas.draw()
                #time.sleep(1)
                
        return x,v1+v2+iter_ops
    else:
        #Create full matrix
        #And solve using a direct solver
        A = create_system(m,h=h)
        shapeF = b.shape
        b = flatten(b)

        x = unflatten(np.linalg.solve(A,b))
        if plot and fig and ax:
            b = unflatten(b)
            r = np.linalg.norm(calculate_residue(x,b,h))

            ax.set_title("Direct: Resolution: {}x{}\n Residue: {}".format(m,m,r))
            ax.imshow(x)
            fig.canvas.draw()
            #time.sleep(1)
        return x , 0

"""
The V-cycle implementation.
Parameters
----------------------------
m : int
    number of nodes
v1 : int
    number of relaxations before coarsening

v2 : int
    number of relaxations after coarsening

rest_type: "injection" or "full_weight"
    Type of restriction function used.

plot: boolean
    True for an visual/interactive mode, showing the current solution
    at each iteration.

random_seed : int
    Seed used to generate a random initial guess (-1 for a initial guess = 0)

Returns
----------------------------

v : 2D Numpy Array
    Final solution

iters: int
    Number of v-cycle iterations

ops : float
    Total Number of Gauss seidel iterations compared to 
    the same number of nodes.
"""
def vcycle(m, min_m = None,v1 = 2, v2=2, atol = 1e-3, rest_type=None,plot=False, random_seed = -1,demo = False):
    if rest_type:
        if rest_type == "injection":
            fullweight = False
        elif rest_type =="full_weight":
            fullweight = True
        else:
            raise Exception("Restriction not supported")
    else:
        fullweight = True

    if plot:
        fig,ax = plt.subplots(1,1)
    else:
        fig,ax = (None,None)
        
    x = np.linspace(0,1,m+2)[1:-1]
    y = np.linspace(0,1,m+2)[1:-1]
    
    h = x[1] - x[0]
    

    if random_seed > 0:
        np.random.seed(random_seed)
        v = np.random.rand(m,m)
    else:
        v = np.zeros((m,m))
        
    f = create_right(x,y)
    
    r = np.linalg.norm(calculate_residue(v,f,h))
    iters = 0
    ops_total = 0 

    while r > atol:

        v,ops = vcycle_iteration(v,f,min_m=min_m, h = h, v1 = v1, v2=v2, fullweight = fullweight,plot = plot, fig=fig, ax = ax)

        r = np.linalg.norm(calculate_residue(v,f,h))

        iters += 1
        ops_total +=ops

        if plot and fig and ax:
            ax.set_title("Iter:{} Resolution: {}x{} \n Residue {}".format(iters,m,m,r))
            ax.imshow(v)
            fig.canvas.draw()
            #time.sleep(1)

        
        if demo and iters == 1:
            break

    return v,iters,ops_total


"""
Check the coarser grid for the initial number of nodes
Parameters
-----------------
m : int
    Number of nodes

Returns
-----------------
The coarser grid for that number of nodes.
"""
def calculate_coarser(m):
    while True:
        mC = ((m-1)/2)+1
        if mC %2==0:
            break
        else:
            m = mC
    return m


"""
Single iteration of multigrid
Parameters
-----------------
x : 2D Numpy Array
    Current Solution
b : 2D Numpy Array
    Right side of the equation
h : float
    Distance between nodes
v0 : int
    Number of v-cycles in each iteration
v1,v2: int,int
    Number of relaxations before and after coarsening in v-cycle

fullweight: boolean
    Uses fullweight restriction if True, otherwise uses injection

plot,ax,fig: boolean, plt.axes, plt.fig
    Parameters for interactive/visual mode

Returns
----------------
x : 2D Numpy Array
    Solution After relaxations.

ops: float
    Number of iterations compared to a Gauss Seidel with the 
    same number of nodes 
"""
def multigrid_iter(x,b,min_m = None, h = None,v0=1, v1 = 2, v2=2, fullweight=True, plot=False, ax=None, fig=None):
    
    if not min_m:
        min_m = 4

    m = len(x)
    if not h:
        h = 1./(m-1)
    
    if m == calculate_coarser(m) or m == min_m:
        for i in range(v0):
            x,ops = vcycle_iteration(x,b,min_m = min_m,h=h,v1=v1,v2=v2,fullweight=fullweight,plot=plot,ax=ax,fig=fig)
        return x,ops
        
    else:
        e = calculate_residue(x,b,h=h)
        eC = restrict(e,fullweight=fullweight)
        xC = np.zeros_like(eC)

        mC = len(xC)
        hC = 1./(mC-1)
        eC,opsC = multigrid_iter(xC,eC,min_m= min_m,h=hC,v0=v0,v1=v1,v2=v2,fullweight=fullweight,plot=plot,ax=ax,fig=fig)
        
        
        x = x + interpolate(eC,m=m)

        for i in range(v0):
            x,ops = vcycle_iteration(x, b, min_m = min_m,h=h , v1=v1,v2=v2,fullweight=fullweight,plot=plot,ax=ax,fig=fig)
        return x,ops+opsC/4.


"""
Full Multigrid Implementation
Parameters:
-----------------------------
m: int
    Number of nodes
v0: int
    Number of v-cycles for each grid

v1,v2: int,int
    Number of relaxations before and after coarsening in vcycle.

plot: boolean
    Interactive/Visual mode

rest_type: "injection" or "full_weight"
    Restriction function

atol: float
    Residual tolerance threshold.

random_seed: int
    Seed to create a random initial guess (-1 to use initial guess = 0)

Returns
-----------------------------
v : 2D Numpy Array
    Final Solution

iters: int
    Number of multigrid iterations

total_ops:
    Number of Gauss seidel iterations with the same amount of nodes.

"""
def full_multigrid(m,min_m = None,v0=1,v1=2,v2=2, plot=False,rest_type=None, atol=1e-3, random_seed= -1, demo=False):
 
    if rest_type:
        if rest_type == "injection":
            fullweight = False
        elif rest_type =="full_weight":
            fullweight = True
        else:
            raise Exception("Restriction not supported")
    else:
        fullweight = True
    
    if plot:
        fig,ax = plt.subplots(1,1)
    else:
        fig,ax = (None,None)
    
    coarser = min_m if min_m else calculate_coarser(m)
    

    h = 1./(m-1)
    
    x = np.linspace(0,1,m+2)[1:-1]
    y = np.linspace(0,1,m+2)[1:-1]    
    
    f = create_right(x,y)
    
    if random_seed > 0:
        np.random.seed(random_seed)
        v = np.random.rand(m,m)
    else:
        v = np.zeros((m,m))
 
    r = np.linalg.norm(calculate_residue(v,f,h))
    iters = 0
    ops_total = 0
    while r > atol:
        v,ops = multigrid_iter(v,f,min_m=min_m,h = h, v0 = v0, v1=v1, v2=v2, fullweight=fullweight, plot=plot, ax=ax, fig=fig)
        r = np.linalg.norm(calculate_residue(v,f,h))
        iters += 1
        ops_total += ops

        if demo and iters == 1:
            break
    return v,iters,ops_total



def experiment1(tol = 1e-8, node_list_initial=range(3,300,2),min_m_list=None, rest_type=None):
    if not min_m_list:
        min_m_list = [3] * len(node_list_initial)

    gauss_iters = []
    vcycle_iters = []
    fmg_iters = []

    vcycle_ops = []
    fmg_ops = []

    gauss_times = []
    vcycle_times = []
    fmg_times = []

    gauss_residues = []
    vcycle_residues = []
    fmg_residues = []

    node_list= []

    print("Starting Experiment")
    for nodes,min_m in zip(node_list_initial,min_m_list):
        try:
            u = create_solution(nodes)
            
            t0_f = time.time()
            v_f,i_f,ops_f=full_multigrid(nodes,min_m = min_m,atol=tol, rest_type=rest_type)
            t1_f = time.time()
            
            t0_v = time.time()    
            v_v,i_v,ops_v=vcycle(nodes,atol=tol,min_m = min_m, rest_type=rest_type)
            t1_v = time.time()

            t0_g = time.time()
            v_g,i_g=gauss_seidel(nodes,atol=tol)
            t1_g = time.time()

            gauss_residues.append(np.linalg.norm(u-v_g))
            gauss_times.append(t1_g-t0_g)
            gauss_iters.append(i_g)

            vcycle_residues.append(np.linalg.norm(u-v_v))
            vcycle_times.append(t1_v-t0_v)
            vcycle_iters.append(i_v)
            vcycle_ops.append(ops_v)
            
            fmg_residues.append(np.linalg.norm(u-v_f))
            fmg_times.append(t1_f-t0_f)
            fmg_iters.append(i_f)
            fmg_ops.append(ops_f)

            node_list.append(nodes)
        except Exception as e:
            pass

    print("Experiment Done.")
    return node_list, gauss_iters,gauss_times,gauss_residues, \
        vcycle_iters,vcycle_times,vcycle_residues,vcycle_ops, \
        fmg_iters,fmg_times,fmg_residues,fmg_ops

def plotExp1Iters(results):
    node_list, gauss_iters,gauss_times,gauss_residues, \
        vcycle_iters,vcycle_times,vcycle_residues,vcycle_ops, \
        fmg_iters,fmg_times,fmg_residues,fmg_ops = results
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))

    ax1.scatter(node_list,gauss_iters,color="red")
    ax1.set_title("Iteraciones Gauss-Seidel")
    ax1.set_xlabel("$n^o$ Nodos")
    ax1.set_ylabel("$n^o$ Iteraciones")

    ax2.scatter(node_list,vcycle_iters,color="green")
    ax2.set_title("Iteraciones V-Cycle")
    ax2.set_xlabel("$n^o$ Nodos")
    ax2.set_ylabel("$n^o$ Iteraciones")


    ax3.scatter(node_list,fmg_iters,color="blue")
    ax3.set_title("Iteraciones Full Multigrid")
    ax3.set_xlabel("$n^o$ Nodos")
    ax3.set_ylabel("$n^o$ Iteraciones")
    plt.show()

def plotExp1Complex(results):
    node_list, gauss_iters,gauss_times,gauss_residues, \
        vcycle_iters,vcycle_times,vcycle_residues,vcycle_ops, \
        fmg_iters,fmg_times,fmg_residues,fmg_ops = results

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,5))
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.plot(node_list,gauss_iters,marker=".", linestyle="-", color="r")
    ax1.plot(node_list,vcycle_ops,marker=".", linestyle="-", color="g")
    ax1.plot(node_list,fmg_ops,marker=".", linestyle="-", color="b")

    ax1.set_title("Complejidad")
    ax1.set_ylabel("$n^o$ iteraciones Gauss-Seidel")
    ax1.set_xlabel("$n^o$ Nodos")
    ax1.legend(["Gauss-Seidel","V-cycle", "Full Multigrid"],loc="upper left")

    ax2.plot(node_list,vcycle_ops,marker=".", linestyle="-", color="g")
    ax2.plot(node_list,fmg_ops,marker=".", linestyle="-", color="b")

    ax2.set_title("Complejidad")
    ax2.set_ylabel("$n^o$ iteraciones Gauss-Seidel")
    ax2.set_xlabel("$n^o$ Nodos")
    ax2.legend(["V-cycle", "Full Multigrid"],loc="upper left")
    plt.show()

def plotExp1Times(results):
    node_list, gauss_iters,gauss_times,gauss_residues, \
        vcycle_iters,vcycle_times,vcycle_residues,vcycle_ops, \
        fmg_iters,fmg_times,fmg_residues,fmg_ops = results

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,5))
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.plot(node_list,gauss_times,marker=".", linestyle="-", color="r")
    ax1.plot(node_list,vcycle_times,marker=".", linestyle="-", color="g")
    ax1.plot(node_list,fmg_times,marker=".", linestyle="-", color="b")

    ax1.set_title("Tiempo")
    ax1.set_ylabel("T [s]")
    ax1.set_xlabel("$n^o$ Nodos")
    ax1.legend(["Gauss-Seidel","V-cycle", "Full Multigrid"],loc="upper left")

    ax2.plot(node_list,vcycle_times,marker=".", linestyle="-", color="g")
    ax2.plot(node_list,fmg_times,marker=".", linestyle="-", color="b")

    ax2.set_title("Tiempo")
    ax2.set_ylabel("T [s]")
    ax2.set_xlabel("$n^o$ Nodos")
    ax2.legend(["V-cycle", "Full Multigrid"],loc="upper left")
    plt.show()

def plotExp1Residues(results):
    node_list, gauss_iters,gauss_times,gauss_residues, \
        vcycle_iters,vcycle_times,vcycle_residues,vcycle_ops, \
        fmg_iters,fmg_times,fmg_residues,fmg_ops = results
    fig,ax1 = plt.subplots(1,1,figsize=(20,5))
    ax1.plot(node_list,gauss_residues,"r")
    ax1.plot(node_list,vcycle_residues,"g")
    ax1.plot(node_list,fmg_residues,"b")

    ax1.set_title("Error")
    ax1.set_ylabel("$u - v$")
    ax1.set_xlabel("$n^o$ Nodos")
    ax1.legend(["Gauss-Seidel","V-cycle", "Full Multigrid"],loc="upper right")
    plt.show()



def experiment2(v0=1,v1=2,v2=2, nodes =65):

    results = []

    for j in range(1,v1+1):
        for k in range(1,v2+1):
            for i in range(1,v0+1):
                result = dict()
                result["fmg"] = dict()
                result["fmg"]["params"] = (nodes,i,j,k) 

                v,i,o = full_multigrid(nodes,v0=i,v1=j,v2=k)
                result["fmg"]["results"] = (v,i,o)
                results.append(result)

        result= dict()
        v,i,o = vcycle(nodes,v1=j,v2=k)
        result["vcycle"] = dict()
        result["vcycle"]["params"] = (nodes,j,k)
        result["vcycle"]["results"] = (v,i,o)
        results.append(result)


    return ((nodes,v0,v1,v2), results)


def plotExp2Results(results,atol=1e-8):
    (nodes,v0,v1,v2), results = results
    
    u = create_solution(nodes)

    error_f = np.zeros((v0,v1,v2))
    iters_f = np.zeros((v0,v1,v2))
    ops_f = np.zeros((v0,v1,v2))


    error_v = np.zeros((v1,v2))
    iters_v = np.zeros((v1,v2))
    ops_v = np.zeros((v1,v2))
    for result in results:
        if result.has_key("fmg"):
            result  = result["fmg"]
            _,i,j,k = result["params"]
            v,iters,o = result["results"]
            iters_f[i-1,j-1,k-1] = iters
            ops_f[i-1,j-1,k-1] = o
            error_f[i-1,j-1,k-1] = np.linalg.norm(u-v)
        else:
            result  = result["vcycle"]
            _,i,j = result["params"]
            v,iters,o = result["results"]
            iters_v[i-1,j-1] = iters
            ops_v[i-1,j-1] = o
            error_v[i-1,j-1] = np.linalg.norm(u-v)

    x_ticks = range(v2+1)
    y_ticks = range(v1+1)

    print("V-Cycle")
    fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,5))           

    im = ax1.imshow(error_v, interpolation='nearest')
    ax1.set_title("Error")
    ax1.set_ylabel(r"$\nu_1$")
    ax1.set_xlabel(r"$\nu_2$")
    fig.colorbar(im, ax = ax1)

    im = ax2.imshow(iters_v, interpolation='nearest')
    ax2.set_title("Iterations")

    ax2.set_ylabel(r"$\nu_1$")
    ax2.set_xlabel(r"$\nu_2$")
    fig.colorbar(im, ax = ax2)

    im = ax3.imshow(ops_v, interpolation='nearest')
    ax3.set_title("Operations")

    ax3.set_ylabel(r"$\nu_1$")
    ax3.set_xlabel(r"$\nu_2$")
    fig.colorbar(im, ax = ax3)

    plt.show()

    for i in range(v0):
        print("FMG v0 = {}".format(i+1))
        fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,5))          
        im = ax1.imshow(error_f[i,:,:],interpolation='nearest')
        ax1.set_title("Error")

        ax1.set_ylabel(r"$\nu_1$")
        ax1.set_xlabel(r"$\nu_2$")
        fig.colorbar(im, ax = ax1)

        im = ax2.imshow(iters_f[i,:,:],interpolation='nearest')
        ax2.set_title("Iterations")
        ax2.set_ylabel(r"$\nu_1$")
        ax2.set_xlabel(r"$\nu_2$")
        fig.colorbar(im, ax = ax2)

        im = ax3.imshow(ops_f[i,:,:],interpolation='nearest')
        ax3.set_title("Operations")
        ax3.set_ylabel(r"$\nu_2$")
        ax3.set_xlabel(r"$\nu_2$")
        fig.colorbar(im, ax=ax3)
        plt.show()

        


