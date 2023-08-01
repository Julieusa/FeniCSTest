# test the difference between V.sub(0) and V.sub(0).collapse
from dolfin import *

mesh = UnitSquareMesh(10,10)
P1 = FiniteElement('P', triangle, 1)
P = P1*P1
V = FunctionSpace(mesh, P)

try:
    view = V.sub(0)
    Function(view)
except RuntimeError: 
    print('Using proper FunctionSpace')
    V1 = V.sub(0).collapse()
    Function(V1)

dim = V1.dim()
assert view.dim() == dim
#In Python, the assert statement is used to continue the execute if the given condition 
# evaluates to True. If the assert condition evaluates to False, then it raises the 
# AssertionError exception with the specified error message.

# Illustrate that V.sub(0) is related to mixed function space by showing that
# some dofs (local) exceed dim
assert all(V1.dofmap().dofs() < dim)

print(any(view.dofmap().dofs() > dim) )

# explain
#a subspace of the original space (in the sense that it expresses everything in terms of 
# the basis elements of the original space; that is, the basis elements have higher dimension
# than the space) and then collapsing as picking a proper basis for my subspace so it becomes
# its own self-contained space