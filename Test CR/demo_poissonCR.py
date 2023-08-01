# Test convergence rate for poisson equation

# LZ: Observe O(h^(r+1))
# .. _demo_poisson_equation:

from dolfin import *
import sympy as sym
from sympy import sin, cos, exp


r=1
x, y = sym.symbols('x[0], x[1]')
u_fcn = 1+x*x+2*y*y
variable_coeff_fcn=1
f = -sym.diff(u_fcn*variable_coeff_fcn,x,2) - sym.diff(u_fcn*variable_coeff_fcn,y,2)

u_fcn = sym.printing.ccode(u_fcn)
f_fcn = sym.printing.ccode(f)
variable_coeff_fcn = sym.printing.ccode(variable_coeff_fcn)
for i in range(1,6): 
    N = int(pow(2,i-1)*4)
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "Lagrange", r)

    def boundary(x, on_boundary):
        return on_boundary
    # Now, the Dirichlet boundary condition can be created using the class
    # :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>`. A
    # :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>` takes three
    # arguments: the function space the boundary condition applies to, the
    # value of the boundary condition, and the part of the boundary on which
    # the condition applies. In our example, the function space is ``V``,
    # the value of the boundary condition (0.0) can represented using a
    # :py:class:`Constant <dolfin.functions.constant.Constant>` and the
    # Dirichlet boundary is defined immediately above. The definition of the
    # Dirichlet boundary condition then looks as follows: ::

    # Define boundary condition
    u00 = Expression(u_fcn,degree=4)
    # u00 = Expression('1+pow(x[0],2)+2*pow(x[1],2)',degree=4)
    bc = DirichletBC(V, u00, boundary)
    variable_coeff=Expression(variable_coeff_fcn,degree=4)

    # Next, we want to express the variational problem.  First, we need to
    # specify the trial function :math:`u` and the test function :math:`v`,
    # both living in the function space :math:`V`. We do this by defining a
    # :py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
    # and a :py:class:`TestFunction
    # <dolfin.functions.function.TrialFunction>` on the previously defined
    # :py:class:`FunctionSpace
    # <dolfin.functions.functionspace.FunctionSpace>` ``V``.
    # 
    # Further, the source :math:`f` and the boundary normal derivative
    # :math:`g` are involved in the variational forms, and hence we must
    # specify these. Both :math:`f` and :math:`g` are given by simple
    # mathematical formulas, and can be easily declared using the
    # :py:class:`Expression <dolfin.functions.expression.Expression>` class.
    # Note that the strings defining ``f`` and ``g`` use C++ syntax since,
    # for efficiency, DOLFIN will generate and compile C++ code for these
    # expressions at run-time.
    # 
    # With these ingredients, we can write down the bilinear form ``a`` and
    # the linear form ``L`` (using UFL operators). In summary, this reads ::

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    # f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    # g = Expression("sin(5*x[0])", degree=2)
    f = Expression(f_fcn,degree=4)
    # f = Constant(-6)
    # f = interpolate(Constant(-6), V)
    a = inner(variable_coeff*grad(u), grad(v))*dx
    # L = f*v*dx + g*v*ds
    L = f*v*dx

    # Now, we have specified the variational forms and can consider the
    # solution of the variational problem. First, we need to define a
    # :py:class:`Function <dolfin.functions.function.Function>` ``u`` to
    # represent the solution. (Upon initialization, it is simply set to the
    # zero function.) A :py:class:`Function
    # <dolfin.functions.function.Function>` represents a function living in
    # a finite element function space. Next, we can call the :py:func:`solve
    # <dolfin.fem.solving.solve>` function with the arguments ``a == L``,
    # ``u`` and ``bc`` as follows: ::

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # The function ``u`` will be modified during the call to solve. The
    # default settings for solving a variational problem have been
    # used. However, the solution process can be controlled in much more
    # detail if desired.
    # 
    # A :py:class:`Function <dolfin.functions.function.Function>` can be
    # manipulated in various ways, in particular, it can be plotted and
    # saved to file. Here, we output the solution to a ``VTK`` file (using
    # the suffix ``.pvd``) for later visualization and also plot it using
    # the :py:func:`plot <dolfin.common.plot.plot>` command: ::
    L2_error  =  errornorm(u, u00, 'L2')
    print(L2_error)
    # # Save solution in VTK format
    # file = File("poisson.pvd")
    # file << u

    # # Plot solution
    # import matplotlib.pyplot as plt
    # plot(u)
    # plt.show()
