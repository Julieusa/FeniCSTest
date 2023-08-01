#
# code for the exact solution case of standard Maxwell's 
# equations solved by Leap-Frog scheme
#
# Modify from LZ's code LFtest_TE_standardMX.py
# TM mode, H is vector
# Final version: tested on FEniCS 2017.2.0 version
# We can observe O(dt^2+h^r)
from dolfin import *
import numpy as np
import sympy as sym
from sympy import sin, cos, exp
from math import pi, log, sqrt
from mshr import *
set_log_active(False) 

dp = 1	# degree of edge element
harray = []  # mesh sizes
Error = []  # errors
for i in range(1,7):
    # Computational Parameters:

    # print( 'hmin=%g hmax=%g:' % (mesh.hmin(), mesh.hmax()))
    # info(mesh)   # print total number of vertices and cells

    # T = 0.1; dt = 0.1/NN; Nt = int(T/dt) # test CR O(dt^2), let dp=2
    NN = int(pow(2,i-1)*4);dt = 1e-3; Nt=1000;T=Nt*dt;h=1/NN;harray.append(h) # test CR for O(h^r)

    # structured mesh: diagnol 'right', 'left', 'crossed', 'right/left' 
    mesh = UnitSquareMesh(NN,NN, 'right')  # number of cells (nx, ny) 
    # if use following mesh, the result is the same
    # mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), NN, NN)
    eps0=1.0; mu0=1.0; 
    mx=1; mgx = mx*pi;    
    ny=1; mgy = ny*pi; 
    meg = sqrt(mgx*mgx + mgy*mgy); 
    # Compute the RHSs
    dpoly = 4		# degree of interpolation polynomial 
    x, y, t = sym.symbols('x[0], x[1], t')

    # exact solution of H=(Hx,Hy)'
    Hx_fcn = mgy/meg*cos(mgx*x)*sin(mgy*y)*sin(meg*t)
    Hy_fcn = -mgx/meg*sin(mgx*x)*cos(mgy*y)*sin(meg*t)
    # exact solution of E
    E_fcn = -cos(mgx*x)*cos(mgy*y)*cos(meg*t)

    # # Use symbolic differentiation to calculate the source term f1, f2
    f1 = eps0*sym.diff(E_fcn, t) - sym.diff(Hy_fcn, x) + sym.diff(Hx_fcn, y) 
    f2x = mu0* sym.diff(Hx_fcn, t) + sym.diff(E_fcn, y) 
    f2y = mu0* sym.diff(Hy_fcn, t) - sym.diff(E_fcn, x) 
    f1_fcn = sym.printing.ccode(f1)
    f2x_fcn = sym.printing.ccode(f2x)
    f2y_fcn = sym.printing.ccode(f2y)

    # construct coefficients for E equation
    cE_1 = dt/eps0
    cH_1 = dt/mu0
    # do this only for 'functions' depending on x,y,t
    E_fcn = sym.printing.ccode(E_fcn)
    Hx_fcn = sym.printing.ccode(Hx_fcn)
    Hy_fcn = sym.printing.ccode(Hy_fcn)

    # Define the function spaces: 
    V = FiniteElement("N1curl", mesh.ufl_cell(), dp)
    U = FiniteElement("DG", mesh.ufl_cell(), dp-1)
    Vs = FunctionSpace(mesh, V)
    Us = FunctionSpace(mesh, U)

    def boundary(x, on_boundary):
        return on_boundary
    # Hbnd_fcn = Expression((Hx_fcn, Hy_fcn), t=0.5*dt, degree=dpoly)
    Hbnd_fcn = Expression((Hx_fcn, Hy_fcn), t=0.0, degree=dpoly)
    Hbc_bnd = DirichletBC(Vs, Hbnd_fcn, boundary)

    # define the functions: iteration starts with n=0
    He = Expression((Hx_fcn, Hy_fcn), t=0, degree=dpoly)
    Ee = Expression(E_fcn, t=0, degree=dpoly)

    # E0 = Expression(E_fcn, t=0.0, degree=dpoly)                 
    # H0 = Expression((Hx_fcn, Hy_fcn), t=0.5*dt, degree=dpoly)  
    # rhs_f1 = Expression(f1_fcn, t=0.5*dt, degree=dpoly)
    # rhs_f2 = Expression((f2x_fcn, f2y_fcn), t=dt, degree=dpoly)

    E0 = Expression(E_fcn, t=-0.5*dt, degree=dpoly)                 
    H0 = Expression((Hx_fcn, Hy_fcn), t=0.0, degree=dpoly)  
    rhs_f1 = Expression(f1_fcn, t=0.0, degree=dpoly)
    rhs_f2 = Expression((f2x_fcn, f2y_fcn), t=0.5*dt, degree=dpoly)

    # setup for FEM
    E0 = interpolate(E0, Us)
    H0 = interpolate(H0, Vs)
    E = interpolate(E0, Us)
    # H = interpolate(H0, Vs)
    # define the variational problem for E
    E = TrialFunction(Us)
    phi = TestFunction(Us)
    a1 = inner(E, phi)*dx
    L1 = inner(E0, phi)*dx + cH_1*inner(curl(H0), phi)*dx + dt*inner(rhs_f1, phi)*dx
    E = Function(Us)

    # define the variational problem for H
    Hn = TrialFunction(Vs)
    psi = TestFunction(Vs)

    a = inner(Hn, psi)*dx
    L = -cH_1*inner(E, curl(psi))*dx + inner(H0, psi)*dx + dt*inner(rhs_f2, psi)*dx 

    Hn = Function(Vs)    # store the solution

    t = 0.0
    from time import *
    s0 = clock()   # measure process time, time(): measure wall time
    for n in range(Nt):    # n starts with 0
        t += dt
        # print ('time step is: %d; t = %g' % (n, t))
    # s0 = clock()   # measure process time, time(): measure wall time
        # iterate the solutions
        if (n>0):
            E0.assign(E)    # assign 'H' to 'H0'
            H0.assign(Hn)

        # rhs_f1.t = t - 0.5*dt
        # rhs_f2.t = t
        rhs_f1.t = t - dt
        rhs_f2.t = t - 0.5*dt
        solve(a1 == L1, E)
        
            # Update H
        # E.assign(project(E0 + dt/eps0*curl(H0)+dt*rhs_f1, Us))
        # if no boundary condition, we can't observe boundary condtion
        # solve(a == L, H)
        solve(a == L, Hn, Hbc_bnd)

    
    e0 = clock()
    print (e0 - s0)   # print out the CPU time spent
    # He.t = T + 0.5*dt 
    # Ee.t = T 
    He.t = T  
    Ee.t = T - 0.5*dt
    errH0 = errornorm(He, Hn, norm_type='l2')
    errE0 = errornorm(Ee, E, norm_type='l2')
    #errE1 = errornorm(Ee, E, norm_type='Hcurl0')

    print('L^2 error of H, t=%g: %-10.6E' % (T, errH0))
    print('L^2 error of E, t=%g: %-10.6E' % (T, errE0))
    #print('H(curl) error of E, t=%g: %-10.6E' % (T, errE1))
    #print('max value of Jp = %g' % norm(Jp,norm_type='l2'))

# Print convergence rates
from math import log as ln
error_types = Error[0].keys()
for error_type in sorted(error_types):
   j = len(Error)
   print ('\nError norm of', error_type)
   for i in range(0, j-1):
       r = ln(Error[i+1][error_type]/Error[i][error_type])/ln(0.5)  # E is a list of errors
       print ('mesh size =%.6f Ei= %-10.6E Ei+1=Ei= %-10.6E r=%.6f' % (harray[i], Error[i][error_type], Error[i+1][error_type], r))