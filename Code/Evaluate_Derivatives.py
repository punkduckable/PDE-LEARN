from typing import Tuple, List;
import torch;

from Network import Neural_Network;

def Evaluate_Derivatives(
        U                         : Neural_Network,
        Highest_Order_Derivatives : int,
        Coords                    : torch.Tensor,
        Device                    : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """ This function evaluates U, its first time derivative, and some of its
    spatial partial derivatives, at each coordinate in Coords.

    Note: This function only works if U is a function of 1 or 2 spatial
    variables.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network we're differentiating.

    Highest_Order_Derivatives: The highest order spatial partial derivatives of
    U that we need to evaluate.

    Coords: A two (1 spatial variable) or three (2 spatial variables) column
    Tensor whose ith row holds the t, x (1 spatial variable) or t, x, y (2
    spatial variables) coordinates of the point where we want to evaluate U
    and its derivatives.

    Device: The device that U is loaded on (either gpu or cpu).

    ----------------------------------------------------------------------------
    Returns:

    This returns a two-element Tuple! If Coords has M rows, then the first
    return argument is an M element Tensor whose ith element holds the value of
    D_{t} U at the ith coordinate.

    The second is a List of tensors, whose kth element holds a tensor that
    holds all of the spatial partial derivatives of U of order k. Suppose that
    Highest_Order_Derivatives = N. The first list item holds a M by 1 tensor
    whose ith entry holds the value of U at the ith Coordinate.

    If Num_Spatial_Dimensions = 1, then the kth list item is a M by 1 tensor
    whose ith element holds D_{x}^k U at the ith coordinate.

    If Num_Spatial_Dimensions = 2, then the kth list item is a M by k+1 tensor
    whose i,j entry holds the value of D_{x}^{k - j} D_{y}^j at the ith
    coordinate. Thus, for example, the 3 columns of the second list item hold
    D_{x}^2 U, D_{x}D_{y} U, and D_{y}^2 U at the coordinates. """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # First, determine the number of spatial variables that U is a function of.
    # This is just the input dimension of U minus 1 (since there's 1 temporal
    # argument)
    Num_Spatial_Dimensions = U.Input_Dim - 1;

    # We have to handle the 1, 2 spatial dimension cases separatly.
    if(Num_Spatial_Dimensions == 1):
        # First, set up the return variables. The tensor Dxn_U will store U and
        # its spatial derivatives.
        Num_Coords : int = Coords.shape[0];
        Dxn_U = [];
        Dxn_U.append(U(Coords).squeeze().view(-1, 1));

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
                    outputs         = Dxn_U[0][:, 0],
                    inputs          = Coords,
                    grad_outputs    = torch.ones_like(Dxn_U[0][:, 0]),
                    retain_graph    = True,
                    create_graph    = True)[0];

        # extract D_{t} U and D_{x} U at each coordinate.
        Dt_U         = Grad_U[:, 0];
        Dxn_U[:, 1]  = Grad_U[:, 1];

        # Compute higher order derivatives
        for i in range(2, Highest_Order_Derivatives + 1):
            # At each coordinate, differentiate D_{x}^{i - 1} U with respect to
            # to t, x. This uses the same process as is described above for
            # Grad_U, but with D_x^{i - 1} U in place of U. We need to create
            # graphs for this so that torch can track this operation when
            # constructing the computational graph for the loss function (which
            # it will use in backpropagation). We also need to retain Grad_U's
            # graph for back-propagation.
            Grad_Dxim1_U = torch.autograd.grad(
                            outputs         = Dxn_U[i-1][:, 1],
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dxn_U[i-1][:, 1]),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Extract D_{x}^{i} U, which is the 1 column of the above Tensor.
            Dxn_U.append(Grad_Dxi_U[:, 1].view(-1, 1));

        return (Dt_U, Dxn_U);

    elif(Num_Spatial_Dimensions == 2):
        # First, set up the return variables. The tensor Dxyn_U is a list whose
        # kth element holds the spatial derivaives of order k of U at the
        # coordinates.
        Num_Coords : int = Coords.shape[0];

        # Evaluate U at the Coords. Note that we need to view the result as a
        # Num_Cooords by 1 tensor.
        Dxyn_U = [];
        Dxyn_U.append(U(Coords).squeeze().view(-1, 1));

        # Set up a tensor to hold the order 1 spatial partial derivatives of U.
        Dxy1_U = torch.empty((Num_Coords, 2), device = Device);

        # Differentiate U with respect to t, x, y at each coordinate. This uses
        # the same process as in the 1 spatial dimension case, except there are
        # three columns. The result is a 3 column Tensor whose (i, 0) entry
        # holds D_{ti} U(t_i, x_i, y_i), (i, 1) entry holds
        # D_{xi} U(t_i, x_i, y_i), and (i, 2) entry holds
        # D_{yi} U(t_i, x_i, y_i).
        Grad_U = torch.autograd.grad(
                    outputs         = Dxyn_U[0][:, 0],
                    inputs          = Coords,
                    grad_outputs    = torch.ones_like(Dxyn_U[0][:, 0]),
                    retain_graph    = True,
                    create_graph    = True)[0];

        # extract D_{t} U,  D_{x} U, and D_{y} U at each coordinate.
        Dt_U         = Grad_U[:, 0];
        Dxy1_U[:, 0] = Grad_U[:, 1];
        Dxy1_U[:, 1] = Grad_U[:, 2];

        # Append the tensor of order 1 spatial partial derivatives to Dxyn_U.
        Dxyn_U.append(Dxy1_U);

        # Compute higher order derivatives
        for i in range(2, Highest_Order_Derivatives + 1):
            # Declare a tensor to hold the order i spatial partial derivatives
            # of U. There are i+1 such derivatives. They are (in order):
            #       D_{x}^i U, D_{x}^{i - 1} D_{y} U,... D_{y}^{i} U
            Dxyi_U = torch.empty((Num_Coords, i + 1), device = Device);

            for j in range(i):
                # To evaluate D_{x}^{i-j} D_{y}^{j} U, we differentiate
                # D_{x}^{i-1-j} D_{y}^{j} U with respect to x. In the case when
                # j = i - 1, we evaluate D_{y}^{i} U by differentiating
                # D_{y}^{i-1} U with respect to y.
                #
                # We need to create graphs for this so that torch can track this
                # operation when constructing the computational graph for the
                # loss function (which it will use in backpropagation). We also
                # need to retain Grad_U's graph for back-propagation.
                Grad_Dxim1mj_Dyj_U = torch.autograd.grad(
                                outputs         = Dxyn_U[i-1][:,j],
                                inputs          = Coords,
                                grad_outputs    = torch.ones_like(Dxyn_U[i-1][:,j]),
                                retain_graph    = True,
                                create_graph    = True)[0];

                # Since
                #   D_{x}^{i - j} D_{y}^{j} U = D_{x} ( D_{i - 1 - j} D_{x}^j U )
                # we want to keep the x derivative in the gradient.
                Dxyi_U[:, j] = Grad_Dxim1mj_Dyj_U[:, 1];

                # If j = i-1, then we also want to keep the y derivative in the
                # gradient, since
                #   D_{y}^{i} U = D_{y} ( D_{y}^{i - 1} U )
                if(j == i - 1):
                    Dxyi_U[:, i] = Grad_Dxim1mj_Dyj_U[:, 2];

            # Append the order i spatial partial derivatives to Dxyn_U.
            Dxyn_U.append(Dxyi_U);

        return (Dt_U, Dxyn_U);
