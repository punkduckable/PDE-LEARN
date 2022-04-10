import torch;
import numpy;
import math;

from Evaluate_Derivatives import Evaluate_Derivatives;
from Network import Neural_Network;
from Mappings import    Col_Number_to_Multi_Index_Class, \
                        Index_to_x_Derivatives, \
                        Index_to_xy_Derivatives_Class;



def Prune_Xi_L2(    Xi                          : torch.Tensor,
                    Threshold                   : float,
                    Index_to_Derivatives,
                    Col_Number_to_Multi_Index,
                    U                           : Neural_Network,
                    Highest_Order_Derivatives   : int,
                    Coords                      : torch.Tensor,
                    Device                      : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function prunes components of Xi. It is basically an experiment
    gone wrong.

    Maybe fill in this description :D """

    # To prune the components of Xi, we first need to scale each one by
    # the 2 norm of the corresponding library term evaluated at the collocation
    # coordinates. To do that, we need to reconstruct the library. First, we
    # need to evaluate the derivatives.
    (Dt_U, DX_U) = Evaluate_Derivatives(U                         = U,
                                        Highest_Order_Derivatives = Highest_Order_Derivatives,
                                        Coords                    = Coords,
                                        Device                    = Device);

    # Initialize Pruned_Xi vector.
    Pruned_Xi : torch.Tensor = torch.empty_like(Xi);

    # Now, loop through the library terms. Calculate the L2 norm of each one,
    # and use this value to decide if we should prune each component.
    Total_Indices : int            = Col_Number_to_Multi_Index.Total_Indices;
    Num_Spatial_Dimensions : int   = U.Input_Dim - 1;
    for k in range(Total_Indices):
        # First, obtain the Multi_Index associated with this column number.
        Multi_Index     = Col_Number_to_Multi_Index(k);
        Num_Sub_Indices = Multi_Index.size;

        # Initialize an array for ith library term. Since we construct this
        # via multiplication, this needs to initialized to a tensor of 1's.
        kth_Lib_Term = torch.ones_like(Dt_U);

        # Now, cycle through the sub-indices in this multi-index to
        # construct the library term. The mechanics of this step depend on if
        # U is a function of one or two spatial variables.
        if(Num_Spatial_Dimensions == 1):
            for j in range(Num_Sub_Indices):
                # First, determine how many derivatives are in the jth term.
                Num_Deriv = Index_to_Derivatives(Multi_Index[j]);

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                kth_Lib_Term = torch.mul(kth_Lib_Term, (DX_U[Num_Deriv][:, 0]).reshape(-1));

        elif(Num_Spatial_Dimensions == 2):
            for j in range(Num_Sub_Indices):
                # First, determine which derivatives are in the jth term.
                Num_xy_Derivs     = Index_to_Derivatives(Multi_Index[j]);
                Num_x_Deriv : int = Num_xy_Derivs[0];
                Num_y_Deriv : int = Num_xy_Derivs[1];
                Num_Deriv : int   = Num_x_Deriv + Num_y_Deriv;

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                kth_Lib_Term = torch.mul(kth_Lib_Term, (DX_U[Num_Deriv][:, Num_y_Deriv]).reshape(-1));

        # Now that we have constructed the kth Library term, we compute its
        # L2 norm.
        Num_Coords                 = kth_Lib_Term.shape[0];
        kth_Lib_Term_2Norm : float = 0;
        for i in range(Num_Coords):
            kth_Lib_Term_2Norm += (kth_Lib_Term[i].item())**2;
        math.sqrt(kth_Lib_Term_2Norm);

        # Now, check if Xi[k]*kth_Lib_Term_2Norm is smaller than the
        # threshold. If so, prune it.
        Abs_Xi_k = abs(Xi[k].item());
        print("Abs_Xi_%d = %f, %dth_Lib_Term_2Norm = %f, Product = %f" % (k, Abs_Xi_k, k, kth_Lib_Term_2Norm, Abs_Xi_k*kth_Lib_Term_2Norm));
        if(Abs_Xi_k*kth_Lib_Term_2Norm < Threshold):
            Pruned_Xi[k] = 0;
        else:
            Pruned_Xi[k] = Xi[k];

    return Pruned_Xi;



def Print_PDE(  Xi                     : torch.Tensor,
                Time_Derivative_Order  : int,
                Num_Spatial_Dimensions : int,
                Index_to_Derivatives,
                Col_Number_to_Multi_Index):
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

    Time_Derivative_Order: We try to solve a PDE of the form (d^n U/dt^n) =
    N(U, D_{x}U, ...). This is the 'n' on the left-hand side of that PDE.

    Num_Spatial_Dimensions: The number of spatial dimensions in the underlying
    data set. We need this to construct the library terms.

    Index_to_Derivatives: If Num_Spatial_Dimensions = 1, then this maps
    sub-index value to a number of x derivatives. If Num_Spatial_Dimensions = 2,
    then this maps a sub-index value to a number of x and y derivatives.

    Col_Number_to_Multi_Index: This maps column numbers (library term numbers)
    to Multi-Indices.

    ----------------------------------------------------------------------------
    Returns:

    Nothing :D """

    if(Time_Derivative_Order == 1):
        print("D_t U = ");
    else:
        print("D_t^%u U = " % Time_Derivative_Order);


    N : int = Xi.numel();
    for k in range(0, N - 1):
        # Fetch the kth component of Xi.
        Xi_k = Xi[k].item();

        # If it's non-zero, fetch the associated multi-Inde
        if(Xi_k == 0):
            continue;
        Multi_Index = Col_Number_to_Multi_Index(k);

        # Cycle through the sub-indices, printing out the associated derivatives
        print("+ %7.4f" % Xi_k, end = '');
        Num_Indices = Multi_Index.size;

        for j in range(0, Num_Indices):
            if  (Num_Spatial_Dimensions == 1):
                Num_x_Deriv : int = Index_to_Derivatives(Multi_Index[j].item());
                if(Num_x_Deriv == 0):
                    print("(U)", end = '');
                else:
                    print("(D_x^%d U)" % Num_x_Deriv, end = '');

            elif(Num_Spatial_Dimensions == 2):
                Num_x_Deriv, Num_y_Deriv = Index_to_Derivatives(Multi_Index[j].item());
                if(Num_x_Deriv == 0):
                    if(Num_y_Deriv == 0):
                        print("(U)", end = '');
                    else:
                        print("(D_y^%d U)" % Num_y_Deriv, end = '');
                elif(Num_y_Deriv == 0):
                    print("(D_x^%d U)" % Num_x_Deriv, end = '');
                else:
                    print("(D_x^%d D_y^%d U)" % (Num_x_Deriv, Num_y_Deriv), end = '');
        print("");

    # Now print out the constant term.
    if(Xi[N - 1] != 0):
        print("+ %7.4f" % Xi[N - 1].item());
