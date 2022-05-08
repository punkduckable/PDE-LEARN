# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

import  torch;
import  numpy;
import  math;
from    typing import List;

from Network    import Neural_Network;
from Derivative import Derivative;
from Term       import Term;



def Print_PDE(  Xi          : torch.Tensor,
                LHS_Term    : Term,
                RHS_Terms   : List[Term]):
    """  This function prints out the PDE encoded in Xi. Suppose that Xi has
    N + 1 components. Then Xi[0] - Xi[N - 1] correspond to PDE library terms,
    while Xi[N] correponds to a constant. Given some k in {0,1,... ,N-1} we
    first map k to a multi-index (using Col_Number_to_Multi_Index). We then map
    each sub-index to a spatial partial derivative of x. We then print out this
    spatial derivative.

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi tensor. If there are N library terms (Col_Number_to_Multi_Index.
    Total_Indices = N), then this should be an N+1 component tensor.

    LHS_Term, RHS_Term: The : We try to learn a PDE of the form
            T_0(U) = Xi_1*T_1(U) + ... + Xi_n*T_N(U).
    where each T_k is a "Term" object. T_0 is the LHS term, T_1, ... , T_N are
    the RHS Terms.

    ----------------------------------------------------------------------------
    Returns:

    Nothing :D """

    print(LHS_Term, end = '');
    print(" = ");
    for i in range(len(RHS_Terms)):
        if(Xi[i] != 0):
            print(" + %7.4f" % Xi[i], end = '');
            print(RHS_Terms[i], end = '');
    print();
