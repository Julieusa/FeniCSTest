
"""
LZ: Leap-Frog scheme, O(dt^2+h^(r+1))
"""

from fenics import *
import numpy as np
set_log_active(False)

mu0=1.0; eps0=1.0;Cv=1.0
m=1;n=1
omega = pi*sqrt(m*m+n*n)
#Define the exact solution by symbolic tool
Ex_code = 'n*pi/omega*cos(m*pi*x[0])*sin(n*pi*x[1])*sin(omega*t)'
Ey_code = '-m*pi/omega*sin(m*pi*x[0])*cos(n*pi*x[1])*sin(omega*t)'
H_code = '-cos(m*pi*x[0])*cos(n*pi*x[1])*cos(omega*t)'

degree=4 # degree for exact solution expression
polyn =4 # degree for trail and test function coef expression

# T = 1.0e-1        # final time for test O(h^(r+1))
# T = 1          # final time for test O(dt)

# Create mesh and define function space
r=2
for i in range(1,6): 
    #test O(h^r)
    N = int(pow(2,i-1)*8)
    nx = ny = N
    h = 1/N
    dt = pow(h, r/2)/32 # time step size
    num_steps = 100    # number of time steps
    T = dt*num_steps

    # #test O(dt)
    # num_steps = int(pow(2,i-1)*4)    # number of time steps
    # dt = T / num_steps # time step size
    # N=num_steps*8
    # nx = ny = N

    # h = 1/N
    # print('dt=',dt)
    # print('h=',h)

    mesh = UnitSquareMesh(nx, ny)
    V = FiniteElement("N2curl", mesh.ufl_cell(), r)
    U = FiniteElement('DG', mesh.ufl_cell(), r-1)
    Vs = FunctionSpace(mesh, V)
    Us = FunctionSpace(mesh, U)
    # W = FunctionSpace(mesh, V*U)

    Esol = Expression((Ex_code,Ey_code), n=n, m=m, omega=omega, t=0.0, degree=degree)
    Hsol = Expression(H_code, n=n, m=m, omega=omega, t=0.5*dt, degree=degree)
    
    # def boundary(x, on_boundary):
    #     return on_boundary

    # Ebc = DirichletBC(Vs, Constant((0.0, 0.0)), boundary)
    # Hbc = DirichletBC(Us, Hsol, boundary)

    # bc = DirichletBC(W.sub(0), Esol, boundary)

    # Define initial value
    E_n = interpolate(Esol,Vs)
    H_n = interpolate(Hsol, Us)
    # E_n = interpolate(Esol, W.sub(0).collapse())
    # H_n = interpolate(Hsol, W.sub(1).collapse())
    vtkfile = File('LF/H0.pvd')
    vtkfile1 = File('LF/E0.pvd')
    vtkfile << E_n
    vtkfile1 << H_n
    Eproj_error_L2 =errornorm(E_n, Esol, 'L2')
    print('Eproj_L2error=', Eproj_error_L2)
    Hproj_error_L2 =errornorm(H_n, Hsol, 'L2')
    print('Hproj_L2error=', Hproj_error_L2)
    # Define variational problem
  
    # (E,H) = TrialFunctions(W)
    # (psi, phi) = TestFunctions(W)

    # Hsolution = Function(W.sub(1))
    H = TrialFunction(Us)
    phi = TestFunction(Us)
    F1 = inner(H, phi)*dx  - inner(H_n, phi)*dx + dt/mu0*inner(curl(E_n),phi)*dx
    a1, L1 = lhs(F1), rhs(F1)

    Hsolution = Function(Us)
    E = TrialFunction(Vs)
    psi = TestFunction(Vs)
    F2 = inner(E,psi)*dx - inner(E_n,psi)*dx - dt/eps0*inner(Hsolution, curl(psi))*dx
    a2, L2 = lhs(F2), rhs(F2)

    Esolution = Function(Vs)

    # Time-stepping
    t = 0.0
    vtkfile2 = File('LF/Hsolution.pvd')
    vtkfile3 = File('LF/Esolution.pvd')
    vtkfile4 = File('LF/Eex.pvd')
    vtkfile5 = File('LF/Hex.pvd')
    for n in range(num_steps):

        # Update current time
        t += dt
        Esol.t = t
        Hsol.t = t-0.5*dt

        # Compute solution
        solve(a1 == L1, Hsolution)
        # (Esolution, Hsolution) = w.split(deepcopy=True)
        solve(a2 == L2, Esolution)

        # Update previous solution
        E_n.assign(Esolution)
        H_n.assign(Hsolution)
    # print(min(E_n.vector().get_local()))
   
    # # Compute error
    vtkfile2 << Hsolution
    vtkfile3 << Esolution
    E_exN = interpolate(Esol, Vs)
    H_exN = interpolate(Hsol, Us)
    ETproj_L2 =errornorm(E_exN, Esol, 'L2')
    print('finalProjError=', ETproj_L2)
    vtkfile4 << E_exN
    vtkfile5 << H_exN
    # Herror_L2 =errornorm(Hsolution, Hsol, 'L2')
    # print('HL2error=', Herror_L2)
    # Eerror_L2 =errornorm(Esolution, Esol, 'L2')
    # print('EL2error=', Eerror_L2)
