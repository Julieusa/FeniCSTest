#
# code for the exact solution case of standard Maxwell's 
# equations solved by Leap-Frog scheme
#
# Dr. Jichun Li, Thu Mar 24 15:08:42 PDT 2022
# 
# Final version: tested on FEniCS 2016.1.0 version
#  same as LF_standardMaxwell.py: just change exact solution!
from dolfin import *
import numpy as np
import sympy as syp
from math import pi, log, sqrt
from mshr import *

# import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# Computational Parameters:
#T = 1e-4
NN = 10
dp = 2	# degree of edge element
#Nt = int(M**(0.5*dp))
#dt = T/Nt


# Define the mesh: unstructured
#circ1 =  Circle(Point(0.0, 0.0), 0.4)
#domain = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
#RESOLUTION = 10   # the larger the finer the mesh
#mesh = generate_mesh(domain, RESOLUTION)
#Nref=1   # refine the mesh several times to test convergence
#for i in range(Nref):
#	mesh=refine(mesh)  

# structured mesh: diagnol 'right', 'left', 'crossed', 'right/left' 
mesh = UnitSquareMesh(NN,NN, 'right')  # number of cells (nx, ny) 
 
print( 'hmin=%g hmax=%g:' % (mesh.hmin(), mesh.hmax()))
info(mesh)   # print total number of vertices and cells
# info(mesh, True)    # print a detailed summary of the object 'mesh'
#plot(mesh, interactive=True)

# Create the triangulation
#n = mesh.num_vertices()
#d = mesh.geometry().dim()
#mesh_coordinates = mesh.coordinates().reshape((n, d))
#triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
#triangulation = tri.Triangulation(mesh_coordinates[:, 0],
#                                  mesh_coordinates[:, 1],
#                                  triangles)

# Plot the mesh
#plt.figure()
#plt.triplot(triangulation)
#plt.savefig('mesh.png')

#dt = int((mesh.hmin())**(0.5*dp))
#T = 1e-6;  dt=1e-7; Nt = int(T/dt)

T = 0.1; dt = 0.1/NN; Nt = int(T/dt)
# Physical Parameters:
# cc = 2.99792458e8; eps0 = 1.0/(cc*cc*mu0); mu0 = 4.0*syp.pi*(1.0e-7); 
#mu0 = 1.25663706212e-6; eps0=8.854187817*1e-12;
#b2=2.418e-31
#b1x=8.082e-17/b2; b2x=1.0/b2; a2x=1.343e-36/b2; a1x=-1.674e-20/b2; 
#a0x=-9.114e-28/b2;
#gama=0.43*1.6022e-13;  # 0.43 meV
#wf = 4.0;  c1=0.0; c2=0.0
#alfa=0.5*gama; wpe = 3.0e12   # 3 THz

eps0=1.0; mu0=1.0; 
mx=1; mgx = mx*syp.pi;    
ny=1; mgy = ny*syp.pi; 
meg = syp.sqrt(mgx*mgx + mgy*mgy); 


# Compute the RHSs
dpoly = 5		# degree of interpolation polynomial 
x, y, t = syp.symbols('x[0], x[1], t')

# exact solution of E=(Ex,Ey)'
Ex_fcn = mgy/meg*syp.cos(mgx*x)*syp.sin(mgy*y)*syp.sin(meg*t)
Ey_fcn = -mgx/meg*syp.sin(mgx*x)*syp.cos(mgy*y)*syp.sin(meg*t)
# exact solution of H
H_fcn = -syp.cos(mgx*x)*syp.cos(mgy*y)*syp.cos(meg*t)



# RHS function f
f1_fcn = 0.0 
f2_fcn = 0.0 


# construct coefficients for E equation
cE_1 = dt/eps0


# do this only for 'functions' depending on x,y,t
vars = [Ex_fcn, Ey_fcn, H_fcn]
vars = [syp.printing.ccode(var) for var in vars]
vars = [var.replace('M_PI', 'pi') for var in vars]
[Ex_fcn, Ey_fcn, H_fcn] = vars

# Define the function spaces: 

V = FunctionSpace(mesh,"Nedelec 1st kind H(curl)",dp)	# space for (Ex, Ey)
U = FunctionSpace(mesh,"DG",dp-1)		# space for H
Vs = FunctionSpace(mesh,"Nedelec 1st kind H(curl)",dp)	
Us = FunctionSpace(mesh,"DG",dp-1)	
W = V*U

def boundary(x, on_boundary):
    return on_boundary
bnd_fcn = Expression((Ex_fcn, Ey_fcn), t=0.0, degree=dpoly)
bc_bnd = DirichletBC(Vs, bnd_fcn, boundary)

# define the functions: iteration starts with n=0
Ee = Expression((Ex_fcn, Ey_fcn), t=0, degree=dpoly)
He = Expression(H_fcn, t=0, degree=dpoly)

H0 = Expression(H_fcn, t=-0.5*dt, degree=dpoly)                 # H^{n-1/2}
E0 = Expression((Ex_fcn, Ey_fcn), t=0, degree=dpoly)  # E^{n}
#rhs_f = Expression((f1_fcn, f2_fcn), t=0.5*dt, degree=dpoly)


# coefficients for E equation
#cE2 = Expression(((cE2_11, cE2_12),(cE2_21, cE2_22)), degree=dpoly)

# coefficients for H
#cH1 = Expression(cH1_fcn, degree = dpoly)

# setup for FEM
H0 = interpolate(H0, Us)
E0 = interpolate(E0, Vs)
H = interpolate(H0, Us)
E = interpolate(E0, Vs)
#rhs_f = interpolate(rhs_f, Vs)


# define the variational problem for E
E = TrialFunction(Vs)
psi = TestFunction(Vs)

a = inner(E, psi)*dx
L = cE_1*inner(H, curl(psi))*dx + inner(E0, psi)*dx  

E = Function(Vs)    # store the solution

t = 0.0
from time import *
s0 = clock()   # measure process time, time(): measure wall time
for n in range(Nt):    # n starts with 0
    t += dt
    print ('time step is: %d; t = %g)' % (n, t))
   # s0 = clock()   # measure process time, time(): measure wall time
	# iterate the solutions
    if (n>0):
        H0.assign(H)    # assign 'H' to 'H0'
        E0.assign(E)
           
	# Update the time of RHS
   # rhs_f.t = t - 0.5*dt
    
        # Update H
    H.assign(project(H0 - dt/mu0*curl(E0), Us))

	# Update E
    #b = assemble(L, tensor=b)   
    #bc_bnd.apply(A, b)
    #solve(A, D.vector(), b)
    solve(a == L, E, bc_bnd)

  
e0 = clock()
print (e0 - s0)   # print out the CPU time spent
Ee.t = T 
He.t = T - 0.5*dt

errE0 = errornorm(Ee, E, norm_type='l2')
errH0 = errornorm(He, H, norm_type='l2')
#errE1 = errornorm(Ee, E, norm_type='Hcurl0')

print('L^2 error of E, t=%g: %-10.6E' % (T, errE0))
print('L^2 error of H, t=%g: %-10.6E' % (T, errH0))
#print('H(curl) error of E, t=%g: %-10.6E' % (T, errE1))
#print('max value of Jp = %g' % norm(Jp,norm_type='l2'))

# calculate \pa_tE errors: 
#Et_hat.t = T 
#E.assign(project((E - E0)/dt, Vs))    # \delta_{tau} E_h^n
#errEt = errornorm(Et_hat, E, norm_type='l2')
#print('L^2 error of E_t, t=%g: %-10.6E' % (T, errEt))


# # # import matplotlib.pyplot as plt
# # # plt.figure()
# # # plot(u0)
# # # plt.show()

# import MyPlot as mp
# mp.MyTriSurf(mesh, E)
