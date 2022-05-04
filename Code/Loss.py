import  numpy;
import  torch;
import  math;
from    typing import Tuple;

from Network  import    Neural_Network;
from Mappings import    x_Derivatives_to_Index, xy_Derivatives_to_Index, \
                        Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, \
                        Col_Number_to_Multi_Index_Class;
from Evaluate_Derivatives import Evaluate_Derivatives;



def Data_Loss(
        U           : Neural_Network,
        Inputs      : torch.Tensor,
        Targets     : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the data loss, which is the mean square
    error between U at the Inputs, and the Targets. To do this, we
    first evaluate U at the data points. At each point (t, x), we then
    evaluate |U(t, x) - U'_{t, x}|^2, where U'_{t,x} is the data point
    corresponding to (t, x). We sum up these values and then divide by the
    number of data points.

    ----------------------------------------------------------------------------
    Arguments:

    U: The neural network which approximates the system response function.

    Inputs: If U is a function of one spatial variable, then this should
    be a two column tensor whose ith row holds the (t, x) coordinate of the
    ith data point. If U is a function of two spatial variables, then this
    should be a three column tensor whose ith row holds the (t, x, y)
    coordinates of the ith data point.

    Targets: If Targets has N rows, then this should be an N element
    tensor whose ith element holds the value of U at the ith data point.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor whose sole entry holds the mean square data loss. """

    # Evaluate U at the data points.
    U_Predict = U(Inputs).squeeze();

    # Evaluate the pointwise square difference of U_Predict and Targets.
    Square_Error = ( U_Predict - Targets ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Coll_Loss(
        U                                   : Neural_Network,
        Xi                                  : torch.Tensor,
        Coll_Points                         : torch.Tensor,
        Time_Derivative_Order               : int,
        Highest_Order_Spatial_Derivatives   : int,
        Index_to_Derivatives,
        Col_Number_to_Multi_Index,
        Device                              : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Let L(U) denote the library matrix (i,j entry is the jth library term
    evaluated at the ith collocation point. Note that we define the last library
    term as the constant function 1). Further, let b(U) denote the vector whose
    ith entry is the time derivative of U at the ith collocation point. Then
    this function returns ||b(N) - L(U)Xi||_2

    ----------------------------------------------------------------------------
    Arguments:

    U: The Neural Network that approximates the solution.

    Xi: A trainable (requires_grad = True) torch 1D tensor. If there are N
    distinct multi-indices, then this should be an N+1 element tensor (the first
    N components are for the library terms represented by multi-indices, the
    final one is for a constant term).

    Coll_Points: A 2 or 3 column tensor of coordinates. If
    Num_Spatial_Dimensions = 1, then the ith row of this tensor holds the
    (t, x) coordinates of the ith collocation point. If Num_Spatial_Dimensions
    = 2, then the ith row of this tensor holds the (t, x, y) coordiante of the
    ith collocation point.

    Time_Derivative_Order: We try to solve a PDE of the form (d^n U/dt^n) =
    N(U, D_{x}U, ...). This is the 'n' on the left-hand side of that PDE.

    Highest_Order_Spatial_Derivatives: The highest order derivatives in our
    library terms. We need to know this to evaluate the spatial partial
    derivatives of Xi and, subsequently, evaluate the library terms.

    Index_to_Derivatives: If Num_Spatial_Dimensions = 1, then this maps a
    sub-index value to a number of x derivatives. If Num_Spatial_Dimensions = 2,
    then this maps a sub-index value to a number of x and y derivatives.

    Col_Number_to_Multi_Index: This maps column numbers (library term numbers)
    to Multi-Indices.

    Device: The device (gpu or cpu) that we train on.

    ----------------------------------------------------------------------------
    Returns:

    A tuple. The first entry of the tuple is a scalar tensor whose lone element
    contains the mean square collocation loss at the Coords. The second is a
    1D tensor whose ith entry holds the PDE residual at the ith collocation
    point. You can safely discard the second return variable if you just want
    to get the loss. """

    # First, determine the number of spatial dimensions. Sice U is a function
    # of t and x, or t and x, y, this is just one minus the input dimension of
    # U.
    Num_Spatial_Dimensions : int = U.Input_Dim - 1;


    # This code behaves differently for 1 and 2 spatial variables.
    if(Num_Spatial_Dimensions == 1):
        # First, acquire the spatial and time derivatives of U.
        (Dtn_U, Dxn_U) = Evaluate_Derivatives(
                            U                                   = U,
                            Time_Derivative_Order               = Time_Derivative_Order,
                            Highest_Order_Spatial_Derivatives   = Highest_Order_Spatial_Derivatives,
                            Coords                              = Coll_Points,
                            Device                              = Device);

        # Construct our approximation to Dtn_U. To do this, we cycle through
        # the columns of the library. At each column, we construct the term
        # and then multiply it by the corresponding component of Xi. We then add
        # the result to a running total.
        Library_Xi_Product = torch.zeros_like(Dtn_U);

        # First, determine the number of columns.
        Total_Indices = Col_Number_to_Multi_Index.Total_Indices;
        for i in range(Total_Indices):
            # First, obtain the Multi_Index associated with this column number.
            Multi_Index     = Col_Number_to_Multi_Index(i);
            Num_Sub_Indices = Multi_Index.size;

            # Initialize an array for ith library term. Since we construct this
            # via multiplication, this needs to initialized to a tensor of 1's.
            ith_Lib_Term = torch.ones_like(Dtn_U);

            # Now, cycle through the indices in this multi-index.
            for j in range(Num_Sub_Indices):
                # First, identify the jth sub-index. This value tells us which
                # entry of Dxn_U holds the derivative corresponding to this
                # sub-index.
                Col : int = Multi_Index[j];

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                ith_Lib_Term = torch.mul(ith_Lib_Term, Dxn_U[Col]);

            # Multiply the ith_Lib_Term by the ith component of Xi and add the
            # result to the Library_Xi product.
            Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(ith_Lib_Term, Xi[i]));

        # Finally, add on the constant term (the final component of Xi!)
        Ones_Col = torch.ones_like(Dtn_U);
        Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(Ones_Col, Xi[Total_Indices]));

        # Now, compute the pointwise residual between Dtn_U and the
        # Library_Xi_Product.
        Residual    : torch.Tensor  = torch.subtract(Dtn_U, Library_Xi_Product);

        # Return the mean square error, as well as the collocation loss at each
        # point.
        return ((Residual**2).mean(), Residual);

    elif(Num_Spatial_Dimensions == 2):
        # First, acquire the spatial and time derivatives of U.
        (Dtn_U, Dxyn_U) = Evaluate_Derivatives(
                            U                                   = U,
                            Time_Derivative_Order               = Time_Derivative_Order,
                            Highest_Order_Spatial_Derivatives   = Highest_Order_Spatial_Derivatives,
                            Coords                              = Coll_Points,
                            Device                              = Device);

        # Construct our approximation to Dtn_U. To do this, we cycle through
        # the columns of the library. At each column, we construct the term
        # and then multiply it by the corresponding component of Xi. We then add
        # the result to a running total.
        Library_Xi_Product = torch.zeros_like(Dtn_U);

        Total_Indices = Col_Number_to_Multi_Index.Total_Indices;
        for i in range(Total_Indices):
            # First, obtain the Multi_Index associated with this column number.
            Multi_Index     = Col_Number_to_Multi_Index(i);
            Num_Sub_Indices = Multi_Index.size;

            # Initialize an array for ith library term. Since we construct this
            # via multiplication, this needs to initialized to a tensor of 1's.
            ith_Lib_Term = torch.ones_like(Dtn_U);

            # Now, cycle through the indices in this multi-index.
            for j in range(Num_Sub_Indices):
                # First, identify the jth sub-index. This value tells us which
                # entry of Dxyn_U holds the derivative corresponding to this
                # sub-index.
                Col : int = Multi_Index[j];

                # Now multiply the ith library term by the corresponding
                # derivative of U.
                ith_Lib_Term = torch.mul(ith_Lib_Term, Dxyn_U[Col]);

            # Multiply the ith_Lib_Term by the ith component of Xi and add the
            # result to the Library_Xi product.
            Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(ith_Lib_Term, Xi[i]));

        # Finally, add on the constant term (the final component of Xi!)
        Ones_Col = torch.ones_like(Dtn_U);
        Library_Xi_Product = torch.add(Library_Xi_Product, torch.mul(Ones_Col, Xi[Total_Indices]));

        # Now, compute the pointwise residual between Dtn_U and the
        # Library_Xi_Product.
        Residual    : torch.Tensor  = torch.subtract(Dtn_U, Library_Xi_Product);

        # Return the mean square error.
        return ((Residual**2).mean(), Residual);



def Lp_Loss(Xi : torch.Tensor, p : float):
    """ This function approximates the L0 norm of Xi using the following
    quantity:
        w_1*|Xi[1]|^2 + w_2*|Xi[2]|^2 + ... + w_N*|Xi[N]|^2
    Where, for each k,
        w_k = 1/max{delta, |Xi[k]|^{p - 2}}.
    (where delta is some small number that ensures we're not dividing by zero!)

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    p: The "p" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        w_1*|Xi[1]|^p + w_2*|Xi[2]|^p + ... + w_N*|Xi[N]|^p
    where N is the number of components of Xi. """

    assert(p > 0 and p < 2)

    # First, square the components of Xi. Also, make a doule precision copy of
    # Xi that is detached from Xi's graph.
    delta : float = .000001;
    Xi_2          = torch.mul(Xi, Xi);
    Xi_Detach     = torch.detach(Xi);

    # Now, define a weights tensor.
    W               = torch.empty_like(Xi_Detach);
    N : int         = W.numel();
    for k in range(N):
        # First, obtain the absolute value of the kth component of Xi, as a float.
        Abs_Xi_k    : float = abs(Xi[k].item());

        # Now, evaluate W[k].
        W_k  = 1./max(delta, Abs_Xi_k**(2 - p));

        # Check for infinity (which can happen, unfortuneatly, if delta is too
        # small). If so, remedy it.
        if(math.isinf(W_k)):
            print("W_k got to infinty");
            print("Abs_Xi_k = %f" % Abs_Xi_k);
            W_k = 0;

        W[k] = W_k;

    # Finally, evaluate the element-wise product of Xi and W[k].
    W_Xi_2 = torch.mul(W, Xi_2);
    return W_Xi_2.sum();



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
