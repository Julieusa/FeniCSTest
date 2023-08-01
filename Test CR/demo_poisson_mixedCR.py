# Poisson equation 
#  LZ: 03/27 I can see the CR is h^r with mixed variational form
# Poisson equation with particular boundary conditions reads:
# Be careful: if change exact solution, we also need to change g. 
# I don't know how to write general sym.diff for g, so this program is not suitable for change exact solution to test CR 
# .. math::
#    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
#                 u &= 0 \quad {\rm on} \ \Gamma_{D}, \\
#                 \nabla u \cdot n &= g \quad {\rm on} \ \Gamma_{N}. \\
from dolfin import *
import sympy as sym
from sympy import sin, cos, exp
# import matplotlib.pyplot as plt

r=2
# uNum= File('u_h.pvd')
# uex = File('uex.pvd')
x, y = sym.symbols('x[0], x[1]')
u_fcn = x(1-x)*y*(1-y)
f = -u_fcn.diff(x,2) - u_fcn.diff(y,2)
u_fcn = sym.printing.ccode(u_fcn)
f_fcn = sym.printing.ccode(f)
for i in range(1,6): 

    N = int(pow(2,i-1)*4)
    # Create mesh
    mesh = UnitSquareMesh(N, N)

    # Define finite elements spaces and build mixed space
    BDM = FiniteElement("BDM", mesh.ufl_cell(), r)
    DG  = FiniteElement("DG", mesh.ufl_cell(), r-1)
    
    W = FunctionSpace(mesh, BDM * DG)
    # Vs = FunctionSpace(mesh, BDM)
    # Us = FunctionSpace(mesh, DG)

    # Define trial and test functions
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)
    # sigma = TrialFunctions(Vs)
    # u = TrialFunctions(Us)
    # tau = TestFunctions(Vs)
    # v = TestFunctions(Us)

    # Define function G such that G \cdot n = g
    class BoundarySource(UserExpression):
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh
            if has_pybind11():
                super().__init__(**kwargs)
        def eval_cell(self, values, x, ufc_cell):
            cell = Cell(self.mesh, ufc_cell.index)
            n = cell.normal(ufc_cell.local_facet)
            g = x[0]*(x[0]-1)
            values[0] = g*n[0]
            values[1] = g*n[1]
        def value_shape(self):
            return (2,)

    G = BoundarySource(mesh, degree=2)

    # Specifying the relevant part of the boundary can be done as for the
    # Poisson demo (but now the top and bottom of the unit square is the
    # essential boundary): ::

    # Define essential boundary on y=0, y=1
    def Nboundary(x):
        return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS
    bc = DirichletBC(W.sub(0), G, Nboundary)

    u_ex = Expression(u_fcn,degree=4)
    u_ex_int = interpolate(u_ex, W.sub(1).collapse())
  
    
    # proj_error = errornorm(u_ex_int, u_ex)
    # print('interpolation error=', proj_error) 

   # Now, all the pieces are in place for the construction of the essential
    # boundary condition: ::

    # Define source function
    # f = Expression('2*x[1]*(1-x[1])+2*x[0]*(1-x[0])', degree=4)
    f = Expression(f_fcn,degree=4)
    # Define variational form
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
    L = - f*v*dx 
    # Compute solution
    # sigma = Function(Vs)
    # u = Function(Vs)
    w = Function(W)
    solve(a == L, w, bc)
    (sigma, u) = w.split(deepcopy=True)
    # uNum << u

    # solve(a == L, w, bc)
    # (sigma, u) = w.split(deepcopy=True)
    # uNum << u

    L2_error  =  errornorm(u, u_ex, 'L2')
    print('L2_error=', L2_error)

    # u_exN = interpolate(u_ex, W.sub(1).collapse())
    # uex << u_exN
