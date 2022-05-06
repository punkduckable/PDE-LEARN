# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

from typing import Tuple, List;
import torch;

from Network    import  Neural_Network;
from Mappings   import  x_Derivatives_to_Index, xy_Derivatives_to_Index, \
                        Num_Sub_Index_Values_1D, Num_Sub_Index_Values_2D;



def Evaluate_Derivatives(
        U                                   : Neural_Network,
        Time_Derivative_Order               : int,
        Highest_Order_Spatial_Derivatives   : int,
        Coords                              : torch.Tensor,
        Device                              : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """ This function evaluates U, its first time derivative, and some of its
    spatial partial derivatives, at each coordinate in Coords.

    Note: This function only works if U is a function of 1 or 2 spatial
    variables.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network we're differentiating.

    Time_Derivative_Order: The order of the desired time derivative.

    Highest_Order_Spatial_Derivatives: The highest order spatial partial
    derivatives of U that we need to evaluate.

    Coords: A two (1 spatial variable) or three (2 spatial variables) column
    Tensor whose ith row holds the t, x (1 spatial variable) or t, x, y (2
    spatial variables) coordinates of the point where we want to evaluate U
    and its derivatives.

    Device: The device that U is loaded on (either gpu or cpu).

    ----------------------------------------------------------------------------
    Returns:

    This returns a two-element Tuple! If Coords has M rows, then the first
    return argument is an M element Tensor whose ith element holds the value of
    D_{t}^{n} U at the ith coordinate, where n = Time_Derivative_Order.

    The second is a List of tensors, whose jth element is the spatial partial
    derivative of U associated with the sub index value j, evaluated at the
    coordinates. """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # First, determine the number of spatial variables that U is a function of.
    # This is just the input dimension of U minus 1 (since there's 1 temporal
    # argument)
    Num_Spatial_Dimensions = U.Input_Dim - 1;

    # We have to handle the 1, 2 spatial dimension cases separatly.
    if(Num_Spatial_Dimensions == 1):
        # First, set up the return variables. We know that each spatial partial
        # derivative of U is associated with a sub-index value. Dxn_U is a list
        # whose jth entry holds the spatial partial derivative of U associated
        # with sub index value j evaluated at the coordinates
        Num_Coords      : int   = Coords.shape[0];
        Num_Sub_Indices : int   = Num_Sub_Index_Values_1D(Highest_Order_Spatial_Derivatives);
        Dxn_U                   = [None]*Num_Sub_Indices;

        # Populate the appropiate entry of Dxn_U with U evaluated at the coords.
        Col : int   = x_Derivatives_to_Index(0);
        Dxn_U[Col]  = U(Coords).view(-1);

        # Differentiate U with respect to t, x at each coordinate. To speed up
        # computations, we do this for all coordinates at once. It's important,
        # however, to understand how this works (because it is not obvious). Let
        # N = Num_Coords. We will focus on how Torch computes the derivative of
        # U with respect to x. Let the Jacobian matrix J be defined as follows:
        #
        #       | (d/dx_0)U(t_0, x_0),  (d/dx_0)U(t_1, x_1),... (d/dx_0)U(t_N, x_N) |
        #       | (d/dx_1)U(t_0, x_0),  (d/dx_1)U(t_1, x_1),... (d/dx_1)U(t_N, x_N) |
        #   J = |  ....                  ....                    ....               |
        #       | (d/dx_N)U(t_0, x_0),  (d/dx_N)U(t_1, x_1),... (d/dx_N)U(t_N, x_N) |
        #
        # Let's focus on the jth column. Here, we compute the derivative of
        # U(t_j, x_j) with respect to x_0, x_1,.... x_N. Since U(t_j, x_j) only
        # depends on x_j, all these derivatives will be zero except for the jth
        # one. Thus, J is a diagonal matrix.
        #
        # When we compute torch.autograd.grad with a non-scalar outputs
        # variable, we need to pass a grad_outputs Tensor which has the same
        # shape. Let v denote the vector we pass as grad outputs. In our case,
        # v is a vector of ones. Torch then computes Jv. Since J is diagonal
        # (by the argument above), the ith component of this product is
        # (d/dx_i)U(t_i, x_i), precisely what we want. Torch does the same thing
        # for derivatives with respect to t.
        #
        # The end result is a 2 column Tensor whose (i, 0) entry holds
        # (d/dt_i)U(t_i, x_i), and whose (i, 1) entry holds (d/dx_i)U(t_i, x_i).
        Grad_U = torch.autograd.grad(
                    outputs         = Dxn_U[Col],
                    inputs          = Coords,
                    grad_outputs    = torch.ones_like(Dxn_U[Col]),
                    retain_graph    = True,
                    create_graph    = True)[0];

        # extract D_{t} U and D_{x} U at each coordinate.
        Dtn_U       = Grad_U[:, 0].view(-1);
        Col : int   = x_Derivatives_to_Index(1);
        Dxn_U[Col]  = Grad_U[:, 1].view(-1);

        # Compute the requested time derivative of U.
        for i in range(2, Time_Derivative_Order + 1):
            # At each coordinate, differentiate D_{t}^{i - 1} U with respect to
            # to t, x. This uses the same process we used for Grad_U (described
            # above), but with D_{t}^{i - 1} U in place of U. We need to create
            # graphs for this so that torch can track this operation when
            # constructing the computational graph for the loss function (which
            # it will use in backpropagation). We also need to retain Grad_U's
            # graph for back-propagation.
            Grad_Dt_U = torch.autograd.grad(
                            outputs         = Dtn_U,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dtn_U),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # The 0 column should contain the ith time derivative of U.
            Dtn_U = Grad_Dt_U[:, 0].view(-1);

        # Compute higher order spatial partial derivatives
        for i in range(2, Highest_Order_Spatial_Derivatives + 1):
            # At each coordinate, differentiate D_{x}^{i - 1} U with respect to
            # to t, x. This uses the same process we used for Grad_U (described
            # above), but with D_{x}^{i - 1} U in place of U. We need to create
            # graphs for this so that torch can track this operation when
            # constructing the computational graph for the loss function (which
            # it will use in backpropagation). We also need to retain Grad_U's
            # graph for back-propagation.
            Col_im1  : int  = x_Derivatives_to_Index(i - 1);
            Col      : int  = x_Derivatives_to_Index(i);

            Grad_Dx_im1_U = torch.autograd.grad(
                            outputs         = Dxn_U[Col_im1],
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dxn_U[Col_im1]),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Extract D_{x}^{i} U, which is the 1 column of the above Tensor.
            Dxn_U[Col] = Grad_Dx_im1_U[:, 1].view(-1);

        return (Dtn_U, Dxn_U);

    elif(Num_Spatial_Dimensions == 2):
        # First, set up the return variables. The tensor Dxyn_U is a list whose
        # kth element holds the spatial derivaives of order k of U at the
        # coordinates.
        Num_Coords      : int   = Coords.shape[0];
        Num_Sub_Indices : int   = Num_Sub_Index_Values_2D(Highest_Order_Spatial_Derivatives);
        Dxyn_U                  = [None]*Num_Sub_Indices;

        # Evaluate U at the Coords. Note that we need to view the result as a
        # Num_Cooords by 1 tensor.
        Col : int   = xy_Derivatives_to_Index(0, 0);
        Dxyn_U[Col] = U(Coords).view(-1);

        # Differentiate U with respect to t, x, y at each coordinate. This uses
        # the same process as in the 1 spatial dimension case, except there are
        # three columns. The result is a 3 column Tensor whose (i, 0) entry
        # holds D_{ti} U(t_i, x_i, y_i), (i, 1) entry holds
        # D_{xi} U(t_i, x_i, y_i), and (i, 2) entry holds
        # D_{yi} U(t_i, x_i, y_i).
        Grad_U = torch.autograd.grad(
                    outputs         = Dxyn_U[Col],
                    inputs          = Coords,
                    grad_outputs    = torch.ones_like(Dxyn_U[Col]),
                    retain_graph    = True,
                    create_graph    = True)[0];

        # extract D_{t} U,  D_{x} U, and D_{y} U at each coordinate.
        Dtn_U            = Grad_U[:, 0].view(-1);

        Col_x : int     = xy_Derivatives_to_Index(1, 0);
        Col_y : int     = xy_Derivatives_to_Index(0, 1);
        Dxyn_U[Col_x]   = Grad_U[:, 1].view(-1);
        Dxyn_U[Col_y]   = Grad_U[:, 2].view(-1);

        # Compute the requested time derivative of U.
        for i in range(2, Time_Derivative_Order + 1):
            # At each coordinate, differentiate D_{t}^{i - 1} U with respect to
            # to t, x. This uses the same process we used for Grad_U (described
            # above), but with D_{t}^{i - 1} U in place of U. We need to create
            # graphs for this so that torch can track this operation when
            # constructing the computational graph for the loss function (which
            # it will use in backpropagation). We also need to retain Grad_U's
            # graph for back-propagation.
            Grad_Dt_U = torch.autograd.grad(
                            outputs         = Dtn_U,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dtn_U),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # The 0 column should contain the ith time derivative of U.
            Dtn_U = Grad_Dt_U[:, 0].view(-1);

        # Compute higher order spatial partial derivatives
        for i in range(2, Highest_Order_Spatial_Derivatives + 1):
            for j in range(i):
                # To evaluate D_{x}^{i-j} D_{y}^{j} U, we differentiate
                # D_{x}^{i-j-1} D_{y}^{j} U with respect to x. In the case when
                # j = i - 1, we evaluate D_{y}^{i} U by differentiating
                # D_{y}^{i-1} U with respect to y.
                #
                # We need to create graphs for this so that torch can track this
                # operation when constructing the computational graph for the
                # loss function (which it will use in backpropagation). We also
                # need to retain Grad_U's graph for back-propagation.
                Col_imjm1_j : int = xy_Derivatives_to_Index(i - j - 1,  j);
                Col_imj_j   : int = xy_Derivatives_to_Index(i - j,      j);

                Grad_Dximjm1_Dyj_U = torch.autograd.grad(
                                outputs         = Dxyn_U[Col_imjm1_j],
                                inputs          = Coords,
                                grad_outputs    = torch.ones_like(Dxyn_U[Col_imjm1_j]),
                                retain_graph    = True,
                                create_graph    = True)[0];

                # Since
                #   D_{x}^{i - j} D_{y}^{j} U = D_{x} ( D_{i - j - 1} D_{x}^j U )
                # we want to keep the x derivative in the gradient.
                Dxyn_U[Col_imj_j] = Grad_Dximjm1_Dyj_U[:, 1].view(-1);

                # If j = i - 1, then we also want to keep the y derivative in the
                # gradient, since
                #   D_{y}^{i} U = D_{y} ( D_{y}^{i - 1} U )
                if(j == i - 1):
                    Col_0_i : int   = xy_Derivatives_to_Index(0, i);
                    Dxyn_U[Col_0_i]  = Grad_Dximjm1_Dyj_U[:, 2].view(-1);

        return (Dtn_U, Dxyn_U);
