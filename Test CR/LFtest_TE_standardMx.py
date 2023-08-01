#
# code for the exact solution case of standard Maxwell's 
# equations solved by Leap-Frog scheme
#
# Modify from Dr. Jichun Li's code LF_standardMaxwell_v2.py
# TE mode, E is vector
# Final version: tested on FEniCS 2017.2.0 version
# We can observe O(dt^2+h^r)


from pyclbr import Function
from dolfin import *
import numpy as np
import sympy as sym
from sympy import sin, cos, exp
from math import pi, log, sqrt
from mshr import *
set_log_active(False) 

# import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# Computational Parameters:
#T = 1e-4
NN = 10
dp = 2	# degree of edge element

# structured mesh: diagnol 'right', 'left', 'crossed', 'right/left' 
mesh = UnitSquareMesh(NN,NN, 'right')  # number of cells (nx, ny) 
 
print( 'hmin=%g hmax=%g:' % (mesh.hmin(), mesh.hmax()))
info(mesh)   # print total number of vertices and cells

T = 0.1; dt = 0.1/NN; Nt = int(T/dt)

eps0=1.0; mu0=1.0; 
mx=1; mgx = mx*pi;    
ny=1; mgy = ny*pi; 
meg = sqrt(mgx*mgx + mgy*mgy); 

# Compute the RHSs
dpoly = 5		# degree of interpolation polynomial 
x, y, t = sym.symbols('x[0], x[1], t')

# exact solution of E=(Ex,Ey)'
Ex_fcn = mgy/meg*cos(mgx*x)*sin(mgy*y)*sin(meg*t)
Ey_fcn = -mgx/meg*sin(mgx*x)*cos(mgy*y)*sin(meg*t)
# exact solution of H
H_fcn = -cos(mgx*x)*cos(mgy*y)*cos(meg*t)

# # Use symbolic differentiation to calculate the source term f1, f2
f1 = eps0*sym.diff(H_fcn, t) + sym.diff(Ey_fcn, x) - sym.diff(Ex_fcn, y) 
f2x = mu0* sym.diff(Ex_fcn, t) - sym.diff(H_fcn, y) 
f2y = mu0* sym.diff(Ey_fcn, t) + sym.diff(H_fcn, x) 
f1_fcn = sym.printing.ccode(f1)
f2x_fcn = sym.printing.ccode(f2x)
f2y_fcn = sym.printing.ccode(f2y)


# construct coefficients for E equation
cE_1 = dt/eps0
cE_2 = dt/mu0
# do this only for 'functions' depending on x,y,t
H_fcn = sym.printing.ccode(H_fcn)
Ex_fcn = sym.printing.ccode(Ex_fcn)
Ey_fcn = sym.printing.ccode(Ey_fcn)

# Define the function spaces: 

V = FiniteElement("N1curl", mesh.ufl_cell(), dp)
U = FiniteElement("DG", mesh.ufl_cell(), dp-1)
Vs = FunctionSpace(mesh, V)
Us = FunctionSpace(mesh, U)

def boundary(x, on_boundary):
    return on_boundary
bnd_fcn = Expression((Ex_fcn, Ey_fcn), t=0.0, degree=dpoly)
bc_bnd = DirichletBC(Vs, bnd_fcn, boundary)
# Hbnd_fcn = Expression(H_fcn, t=0.0, degree=dpoly)
# Hbc_bnd = DirichletBC(Us, Hbnd_fcn, boundary)
# define the functions: iteration starts with n=0
Ee = Expression((Ex_fcn, Ey_fcn), t=0, degree=dpoly)
He = Expression(H_fcn, t=0, degree=dpoly)

H0 = Expression(H_fcn, t=-0.5*dt, degree=dpoly)                 # H^{n-1/2}
E0 = Expression((Ex_fcn, Ey_fcn), t=0, degree=dpoly)  # E^{n}
rhs_f1 = Expression(f1_fcn, t=0, degree=dpoly)
rhs_f2 = Expression((f2x_fcn, f2y_fcn), t=0.5*dt, degree=dpoly)

# setup for FEM
H0 = interpolate(H0, Us)
E0 = interpolate(E0, Vs)
H = interpolate(H0, Us)
# E = interpolate(E0, Vs)
#rhs_f = interpolate(rhs_f, Vs)

# define the variational problem for H
H = TrialFunction(Us)
phi = TestFunction(Us)

a1 = inner(H, phi)*dx
L1 = -cE_2*inner(curl(E0), phi)*dx + inner(H0, phi)*dx + dt*inner(rhs_f1, phi)*dx 
H = Function(Us)
# Hsolution = Function(Us)    # store the solution

# define the variational problem for E
En = TrialFunction(Vs)
psi = TestFunction(Vs)

# a = inner(En, psi)*dx
# L = cE_1*inner(Hsolution, curl(psi))*dx + inner(E0, psi)*dx + dt*inner(rhs_f2, psi)*dx 

a = inner(En, psi)*dx
L = cE_1*inner(H, curl(psi))*dx + inner(E0, psi)*dx  + dt*inner(rhs_f2, psi)*dx # define H as some known value, change it later
En = Function(Vs)    # store the solution

t = 0.0

from time import *
s0 = clock()   # measure process time, time(): measure wall time
for n in range(Nt):    # n starts with 0
    t += dt
    # print ('time step is: %d; t = %g' % (n, t))
   # s0 = clock()   # measure process time, time(): measure wall time
	# iterate the solutions
    if (n>0):
        H0.assign(H)    # assign 'H' to 'H0'
        E0.assign(En)
    
        # Update H
    # Hbnd_fcn.t=t
    solve(a1 == L1, H)#, Hbc_bnd)
    
    # H.assign(project(H0 - dt/mu0*curl(E0)+dt*rhs_f1, Us))

    # if no boundary condition, we can't observe boundary condtion
    # solve(a == L, E)
    solve(a == L, En, bc_bnd)

  
e0 = clock()
print (e0 - s0)   # print out the CPU time spent
Ee.t = T 
He.t = T - 0.5*dt

errE0 = errornorm(Ee, En, norm_type='l2')
errH0 = errornorm(He, H, norm_type='l2')
#errE1 = errornorm(Ee, E, norm_type='Hcurl0')

print('L^2 error of E, t=%g: %-10.6E' % (T, errE0))
print('L^2 error of H, t=%g: %-10.6E' % (T, errH0))
#print('H(curl) error of E, t=%g: %-10.6E' % (T, errE1))
#print('max value of Jp = %g' % norm(Jp,norm_type='l2'))

