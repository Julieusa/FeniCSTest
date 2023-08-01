#
# code for the exact solution case of standard Maxwell's 
# equations with variable coefficient solved by Leap-Frog scheme
#
# Modify from LZ's code LFtest_TM_standardMX.py
# TM mode, H is vector
# Final version: tested on FEniCS 2017.2.0 version
# We can observe O(dt^2+h^r)
from ast import Expression
from dolfin import *
import numpy as np
import sympy as sym
from sympy import sin, cos, exp
from math import pi, log, sqrt
from mshr import *
set_log_active(False) 

# Computational Parameters:
# T = 1e-4
NN = 20
dp = 2	# degree of edge element

# structured mesh: diagnol 'right', 'left', 'crossed', 'right/left' 
mesh = UnitSquareMesh(NN,NN, 'right')  # number of cells (nx, ny) 
# if use following mesh, the result is the same
# mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), NN, NN)
 
print( 'hmin=%g hmax=%g:' % (mesh.hmin(), mesh.hmax()))
info(mesh)   # print total number of vertices and cells

T = 0.1; dt = 0.1/NN; Nt = int(T/dt) # test CR O(dt^2), let dp=2
# dt = 1e-5; Nt=1000; T=Nt*dt # test CR for O(h^r)

eps0=1.0; mu0=1.0;
omega =2
omega_t = 2 
# Compute the RHSs
dpoly = 5		# degree of interpolation polynomial 
x, y, t = sym.symbols('x[0], x[1], t')

# exact solution of E=(Ex,Ey)'
Hx_fcn = cos(omega*pi*x)*sin(omega*pi*y)*sin(omega_t*pi*t)
Hy_fcn = -sin(omega*pi*x)*cos(omega*pi*y)*sin(omega_t*pi*t)
# exact solution of H
E_fcn = -cos(omega*pi*x)*cos(omega*pi*y)*cos(omega_t*pi*t)

varCoef_fcn = x + y
# # Use symbolic differentiation to calculate the source term f1, f2
f1 = eps0*sym.diff(E_fcn, t) - sym.diff(Hy_fcn, x) + sym.diff(Hx_fcn, y) 
f2x = mu0* sym.diff(Hx_fcn, t) + sym.diff(E_fcn*varCoef_fcn, y) 
f2y = mu0* sym.diff(Hy_fcn, t) - sym.diff(E_fcn*varCoef_fcn, x) 
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
varCoef_fcn = sym.printing.ccode(varCoef_fcn)
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
varCoef = Expression(varCoef_fcn, degree=dpoly)
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
L = -cH_1*varCoef*inner(E, curl(psi))*dx + inner(H0, psi)*dx + dt*inner(rhs_f2, psi)*dx 

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

