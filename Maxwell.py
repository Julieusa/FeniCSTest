# Standard Maxwell's equation without PML
# we can observe the reflection 
# LZ, FEniCS 2017.2.0
from dolfin import *
from math import log
import sympy as syp
import numpy as np

# % constants
mu0=1.256e-6; #free space permeability
Cv=3e8;eps0=1/(Cv*Cv*mu0); #free space permittivity
sw_f0 = 1.5e9
Source = Expression('sin(2*pi*sw_f0*t)', sw_f0=sw_f0, t=0.0, degree=2)

# %number of PML layers
nPML=12;

T = 2e-8
Nt =20000;dt = T/Nt
N = 100
Nx=2*nPML+N; Ny=2*nPML+N; 

#vacuum domain
ax0 = ay0 = 0.0
bx0 = by0 = 2

# vacuum domain + PML
ax = ax0 - nPML*(bx0-ax0)/Nx
bx = bx0 + nPML*(bx0-ax0)/Nx
ay = ay0 - nPML*(by0-ay0)/Ny
by = by0 + nPML*(by0-ay0)/Ny

mesh = RectangleMesh(Point(ax, ay), Point(bx, by), Nx, Ny)
meshfile = File("Maxwellresult/mesh.pvd") #put the mesh data to a new folder
meshfile << mesh

V = FiniteElement("N2curl", mesh.ufl_cell(), 1)
U = FiniteElement("DG", mesh.ufl_cell(), 0)
Vs = FunctionSpace(mesh, V)
Us = FunctionSpace(mesh, U)

def boundary(x, on_boundary):
    return on_boundary

Hbc = DirichletBC(Vs, Constant((0.0, 0.0)), boundary)
Ebc = DirichletBC(Us, Constant(0.0), boundary)

tol = 1e-2
# domain for source
scr_mark = []     # creat a list,marker for source
phy_mark = []     # marker for physical domain
dm = Us.dofmap()   # mapping of degrees of freedom
xc = 1.00
yc = 1.00
for c in cells(mesh):
    x = c.midpoint().x()  # the midpoint of x??
    y = c.midpoint().y()
    if (x-xc)**2+(y-yc)**2<=tol*tol:
        scr_mark.extend(dm.cell_dofs(c.index())) # extend list with new content from the end
    if between(x, (ax0, bx0)) and between(y, (ay0, by0)):
	    phy_mark.extend(dm.cell_dofs(c.index()))
# Define functions for solutions at previous and current time steps
E0 = Constant(0.0)
H0 = Constant((0.0, 0.0))
E0 = interpolate(E0, Us)
H0 = interpolate(H0, Vs)

E = interpolate(E0, Us)
H = interpolate(H0, Vs)

E = TrialFunction(Us)
psi = TestFunction(Us)
F1 = -eps0*inner(E,psi)*dx + eps0*inner(E0,psi)*dx + inner(dt*curl(H0), psi)*dx
a1, L1 = lhs(F1), rhs(F1)
A1 = assemble(a1)
b1 = None
Esolution = Function(Us)

H = TrialFunction(Vs)
phi = TestFunction(Vs)
F2 = -mu0*inner(H, phi)*dx  + mu0*inner(H0, phi)*dx - dt*inner(Esolution,curl(phi))*dx
a2, L2 = lhs(F2), rhs(F2)
A2 = assemble(a2)
b2 = None
Hsolution = Function(Vs)
# Time-stepping
vtkfile = File('Maxwellresult/Hsolution.pvd')
vtkfile1 = File('Maxwellresult/Esolution.pvd')
t = 0
for n in range(Nt):

    print( 'time step is: %d (total steps: %d)' % (n, Nt))
    t += dt
    b1 = assemble(L1, tensor=b1)
    Ebc.apply(A1, b1)
    solve(A1, Esolution.vector(), b1)
    
    b2 = assemble(L2, tensor=b2)
    Hbc.apply(A2, b2)
    solve(A2, Hsolution.vector(), b2)

    Source.t = t

    src_fcn = interpolate(Source, Us)
    Esolution.vector()[scr_mark] = src_fcn.vector()[scr_mark] # source as initial condition
    if (n>0):
        E0.assign(Esolution)
        H0.assign(Hsolution)

    if (n%100==0):
        vtkfile << (Hsolution, t)
        vtkfile1 << (Esolution, t)
