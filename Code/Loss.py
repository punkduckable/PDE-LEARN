import numpy;
import torch;

from Network import Neural_Network;
from Mappings import    Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, Col_Number_to_Multi_Index_Class;
from Evaluate_Derivatives import Evaluate_Derivatives;



def Data_Loss(
        U           : Neural_Network,
        Data_Points : torch.Tensor,
        Data_Values : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the data loss, which is the mean square
    error between U at the data_points, and the data_values. To do this, we
    first evaluate U at the data points. At each point (t, x), we then
    evaluate |U(t, x) - U'_{t, x}|^2, where U'_{t,x} is the data point
    corresponding to (t, x). We sum up these values and then divide by the
    number of data points.

    ----------------------------------------------------------------------------
    Arguments:

    U: The neural network which approximates the system response function.

    Data_Points: If U is a function of one spatial variable, then this should
    be a two column tensor whose ith row holds the (t, x) coordinate of the
    ith data point. If U is a function of two spatial variables, then this
    should be a three column tensor whose ith row holds the (t, x, y)
    coordinates of the ith data point.

    Data_Values: If Data_Points has N rows, then this should be an N element
    tensor whose ith element holds the value of U at the ith data point.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor whose sole entry holds the mean square data loss. """

    # Evaluate U at the data points.
    U_Predict = U(Data_Points).squeeze();

    # Evaluate the pointwise square difference of U_Predict and Data_Values.
    Square_Error = ( U_Predict - Data_Values ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Coll_Loss(
        U           : Neural_Network,
        Xi          : torch.Tensor,
        Coll_Points : torch.Tensor,
        Highest_Order_Derivatives : int,
        Index_to_Derivatives,
        Col_Number_to_Multi_Index,
        Device      : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ Describe me :D

    Xi should be a 1D Tensor. If there are N distinct multi-indices, then this
    should be an N+1 element tensor (the first N components are for the library
    terms, the final one is for a constant term). """

    # First, determine the number of spatial dimensions. Sice U is a function
    # of t and x, or t and x, y, this is just one minus the input dimension of
    # U.
    Num_Spatial_Dimensions : int = U.Input_Dim - 1;


    # This code behaves differently for 1 and 2 spatial variables.
    if(Num_Spatial_Dimensions == 1):
        # First, acquire the spatial and time derivatives of U.
        (Dt_U, Dx_U) = Evaluate_Derivatives(
                            U      = U,
                            Highest_Order_Derivatives = Highest_Order_Derivatives,
                            Coords = Coll_Points,
                            Device = Device);

        # Construct our approximation to Dt_U. To do this, we cycle through
        # the columns of the library. At each column, we construct the term
        # and then multiply it by the corresponding component of Xi. We then add
        # the result to a running total.
        Library_Xi_Product = torch.zeros_like(Dt_U);

        # First, determine the number of columns.
        Total_Indices = Col_Number_to_Multi_Index.Total_Indices;
        for i in range(Total_Indices):
            # First, obtain the Multi_Index associated with this column number.
            Multi_Index     = Col_Number_to_Multi_Index(i);
            Num_Sub_Indices = Multi_Index.size;

            # Initialize an array for ith library term. Since we construct this
            # via multiplication, this needs to initialized to a tensor of 1's.
            ith_Lib_Term = torch.ones_like(Dt_U);

            # Now, cycle through the indices in this multi-index.
            for j in range(Num_Sub_Indices):
                # First, determine how many derivatives are in the jth term.
                Num_Deriv = Index_to_Derivatives(Multi_Index[j]);

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                ith_Lib_Term = torch.mul(ith_Lib_Term, (Dx_U[Num_Deriv][:, 0]).reshape(-1));

            # Multiply the ith_Lib_Term by the ith component of Xi and add the
            # result to the Library_Xi product.
            Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(ith_Lib_Term, Xi[i]));

        # Finally, add on the constant term (the final component of Xi!)
        Ones_Col = torch.ones_like(Dt_U);
        Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(Ones_Col, Xi[Total_Indices]));

        # Now, compute the pointwise square error between Dt_U and the
        # Library_Xi_Product.
        Square_Error = ( Dt_U - Library_Xi_Product )**2;

        # Return the mean square error.
        return Square_Error.mean();

    else: # Num Spatial dimensions == 2.
        # First, acquire the spatial and time derivatives of U.
        (Dt_U, Dxy_U) = Evaluate_Derivatives(
                            U      = U,
                            Highest_Order_Derivatives = Highest_Order_Derivatives,
                            Coords = Coll_Points,
                            Device = Device);

        # Construct our approximation to Dt_U. To do this, we cycle through
        # the columns of the library. At each column, we construct the term
        # and then multiply it by the corresponding component of Xi. We then add
        # the result to a running total.
        Library_Xi_Product = torch.zeros_like(Dt_U);

        Total_Indices = Col_Number_to_Multi_Index.Total_Indices;
        for i in range(Total_Indices):
            # First, obtain the Multi_Index associated with this column number.
            Multi_Index     = Col_Number_to_Multi_Index(i);
            Num_Sub_Indices = Multi_Index.size;

            # Initialize an array for ith library term. Since we construct this
            # via multiplication, this needs to initialized to a tensor of 1's.
            ith_Lib_Term = torch.ones_like(Dt_U);

            # Now, cycle through the indices in this multi-index.
            for j in range(Num_Sub_Indices):
                # First, determine how many derivatives are in the jth term.
                Num_xy_Derivs = Index_to_Derivatives(Multi_Index[j]);
                Num_x_Deriv : int = Num_xy_Derivs[0];
                Num_y_Deriv : int = Num_xy_Derivs[1];
                Num_Deriv = Num_x_Deriv + Num_y_Deriv;

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                ith_Lib_Term = torch.mul(ith_Lib_Term, (Dxy_U[Num_Deriv][:, Num_y_Deriv]).reshape(-1));

            # Multiply the ith_Lib_Term by the ith component of Xi and add the
            # result to the Library_Xi product.
            Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(ith_Lib_Term, Xi[i]));

        # Finally, add on the constant term (the final component of Xi!)
        Ones_Col = torch.ones_like(Dt_U);
        Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(Ones_Col, Xi[Total_Indices]));

        # Now, compute the pointwise square error between Dt_U and the
        # Library_Xi_Product.
        Square_Error = ( Dt_U - Library_Xi_Product )**2;

        # Return the mean square error.
        return Square_Error.mean();



def Lp_Loss(Xi : torch.Tensor, p : float):
    """ This function approximates the L0 norm of Xi by summing the pth powers
    of the components of Xi. In particular, it evaluates
        |Xi[1]|^p + |Xi[2]|^p + ... + |Xi[N]|^p

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    p: The "p" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        |Xi[1]|^p + |Xi[2]|^p + ... + |Xi[N]|^p
    where N is the number of components of Xi. """

    assert(p > 0)

    # First, take the absolute value of the components of Xi.
    Abs_Xi = torch.abs(Xi);

    # Now, define a "Power Tensor". This tensor has the same shape as Abs_Xi.
    # If Abs_Xi[i] is zero, then Power_Tensor[i] = 1. Otherwise,
    # Power_Tensor[i] = p. The reason we do this is to avoid trying to evaluate
    # p/x^(1 - p) when x = 0.
    Power_Tensor = torch.empty_like(Abs_Xi);
    for i in range(Abs_Xi.numel()):
        if(Abs_Xi[i] == 0):
            Power_Tensor[i] = 1;
        else:
            Power_Tensor[i] = p;


    # Now, raise the components to the pth power and sum.
    Abs_Xi_p = torch.pow(Abs_Xi, Power_Tensor);
    return Abs_Xi_p.sum();



def L0_Approx_Loss(Xi : torch.Tensor, s : float):
    """ This function returns an approximation to the L0 norm of Xi. Notice that
    if x is a real number then,
            lim_{s -> 0} 1 - exp(-x^2/s) = { 0    if x = 0
                                           { 1    if x != 0
    Thus, if we choose s to be small, then
            N - ( exp(-Xi_1^2/s^2) + exp(-Xi_2^2/s^2) + ... + exp(-Xi_N^2/s^2) )
    will approximate the L0 norm of x. This function evaluates the expression
    above.

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    s: The "s" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        N - ( exp(-Xi_1^2/s^2) + exp(-Xi_2^2/s^2) + ... + exp(-Xi_N^2/s^2) )
    where N is the number of components of Xi. """

    # s must be positive for the following to work.
    assert(s > 0);

    # Component-wise evaluate exp(-Xi^2/s)
    Xi2_d_s2     = torch.div(torch.square(Xi), s*s);
    Exp_Xi2_d_s2 = torch.exp(torch.mul(Xi2_d_s2, -1));

    # Take the sum and subtract N from it.
    N : int = Xi.numel();
    return N - Exp_Xi2_d_s2.sum();
