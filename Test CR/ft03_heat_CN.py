
"""
LZ: Crank-Nicolson scheme, O(dt^2+h^(r+1))

FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t 
  this solution after 2nd derative no t, so we can observe o(h^(r+1)) without restriction of dt
  f = beta - 2 - 2*alpha
"""
# try to use u = 1 + x^2 + alpha*y^2 + \beta*t^3 to test CR for dt
from __future__ import print_function
from fenics import *
import numpy as np
set_log_active(False)
T = 1.0e-1        # final time for test O(h^(r+1))
# T = 1          # final time for test O(dt)
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta
# Create mesh and define function space
r=1
for i in range(1,6): 
#   #test O(h^(r+1))
#   num_steps = 1000    # number of time steps
#   dt = T / num_steps # time step size
#   N = int(pow(2,i-1)*8)
#   nx = ny = N

  #test O(dt)
  num_steps = int(pow(2,i-1)*4)    # number of time steps
  dt = T / num_steps # time step size
  N=num_steps
  nx = ny = N

  h = 1/N
  print('dt=',dt)
  print('h=',h)
  mesh = UnitSquareMesh(nx, ny)
  V = FunctionSpace(mesh, 'P', r)

  # Define boundary condition
  order_t = 3
  u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*pow(t,order_t)', order_t=order_t,
                  degree=4, alpha=alpha, beta=beta, t=0)

  def boundary(x, on_boundary):
      return on_boundary

  bc = DirichletBC(V, u_D, boundary)

  # Define initial value
  u_n = interpolate(u_D, V)
  #u_n = project(u_D, V)
  Uproj_error_L2 =errornorm(u_n, u_D, 'L2')
  print('Uproj_L2error=', Uproj_error_L2)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
#   f = Constant(beta - 2 - 2*alpha)
  f0 = Expression('order_t*beta*pow(t,order_t-1) - 2 - 2*alpha', order_t=order_t, beta=beta, alpha=alpha, t=0, degree=4)
  f1 = Expression('order_t*beta*pow(t,order_t-1) - 2 - 2*alpha', order_t=order_t, beta=beta, alpha=alpha, t=dt, degree=4)
  f_middle = Expression('order_t*beta*pow(t,order_t-1) - 2 - 2*alpha', order_t=order_t, beta=beta, alpha=alpha, t=0.5*dt, degree=4)
#   F = inner(u-u_n,v)*dx + dt*0.5*inner(grad(u+u_n), grad(v))*dx - 0.5*dt*inner(f0+f1,v)*dx
#   a, L = lhs(F), rhs(F)
 
  F = inner(u-u_n,v)*dx + dt*0.5*inner(grad(u+u_n), grad(v))*dx - dt*inner(f_middle,v)*dx
  a, L = lhs(F), rhs(F)
  
  # Time-stepping
  u = Function(V)
  t = 0
  for n in range(num_steps):

      # Update current time
      t += dt
      u_D.t = t
      f0.t = t-dt
      f1.t = t
      f_middle.t = t-0.5*dt

      # Compute solution
      solve(a == L, u, bc)

      # Update previous solution
      u_n.assign(u)
      
  # print(max(u.vector().get_local()))
  # Compute error at vertices. max error
  # u_e = interpolate(u_D, V)
  # error = np.abs(u_e.vector().array() - u.vector().array()).max()
  # print('t = %.2f: error = %.3g' % (t, error))

#   error_L2 =errornorm(u, u_D, 'L2')
#   print('L2error=', error_L2)
