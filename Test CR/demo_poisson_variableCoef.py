"""
FEniCS tutorial demo program:
The Poisson equation with a variable coefficient.

-div(p*grad(u)) = f on the unit square.
u = u0 on x=0,
u0 = u = 1 + x^2 + 2y^2, p = x + y, f = -8x - 10y.
 
"""
'''
# LZ: 04/19/2022 test convergence rate
# use given function
from dolfin import *
import sympy as sym
from sympy import sin, cos, exp
# Create mesh and define function space
N = 20
r=1

mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=4)

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

u0_boundary = DirichletBoundary()
bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = Expression('x[0] + x[1]', degree=4)
f = Expression('-8*x[0] - 10*x[1]', degree=4)
a = inner(p*nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)
L2_error  =  errornorm(u, u0, 'L2')
print(L2_error)
'''
# use symbolic calculation for any exact solution
from ast import Expression
from dolfin import *
import sympy as sym
from sympy import sin, cos, exp
# Create mesh and define function space
N = 20
r=1

mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
x, y = sym.symbols('x[0], x[1]')
u_fcn = 1+2*x*x+2*y*y
variable_coeff_fcn= x**2 + y**2
du_dx = sym.diff(u_fcn,x)
du_dy = sym.diff(u_fcn,y)
f = -sym.diff(variable_coeff_fcn*du_dx,x) - sym.diff(variable_coeff_fcn*du_dy,y)

u_fcn = sym.printing.ccode(u_fcn)
f_fcn = sym.printing.ccode(f)
variable_coeff_fcn = sym.printing.ccode(variable_coeff_fcn)

u0 = Expression(u_fcn, degree=4)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

u0_boundary = DirichletBoundary()
bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
# p = Expression('x[0] + x[1]', degree=4)
# f = Expression('-8*x[0] - 10*x[1]', degree=4)
f = Expression(f_fcn, degree=4)
variable_coeff = Expression(variable_coeff_fcn, degree=4)
a = inner(variable_coeff*nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)
L2_error  =  errornorm(u, u0, 'L2')
print(L2_error)