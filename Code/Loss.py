# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

import  numpy;
import  torch;
import  math;
from    typing import Tuple, List;

from Derivative             import Derivative;
from Term                   import Term;
from Network                import Neural_Network;
from Evaluate_Derivatives   import Derivative_From_Derivative;



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
        Derivatives                         : List[Derivative],
        LHS_Term                            : Term,
        RHS_Terms                           : List[Term],
        Device                              : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Let L(U) denote the library matrix (i,j entry is the jth RHS term
    evaluated at the ith collocation point. Further, let b(U) denote the vector
    whose ith entry is the LHS Term at the ith collocation point. Then
    this function returns ||b(U) - L(U)Xi||_2

    ----------------------------------------------------------------------------
    Arguments:

    U: The Neural Network that approximates the solution.

    Xi: A trainable (requires_grad = True) torch 1D tensor. If there are N
    RHS_Terms, this should be an N element vector.

    Coll_Points: B by n column tensor, where B is the number of coordinates and
    n is the dimension of the problem domain. The ith row of Coll_Points should
    hold the components of the ith coordinate.

    Derivatives: We try to solve a PDE of the form
            T_0(U) = Xi_1*T_1(U) + ... + Xi_n*T_N(U).
    where each T_k is a "Term" of the form
            T_k(U) = (D_1 U)^{p(1)} ... (D_m U)^{p(m)}
    where each D_j is a derivative operator and p(j) >= 1. Derivatives is a list
    that contains each D_j's in each term in the equation above. This list
    should be ordered according to the Derivatives' orders (see Derivative
    class).

    LHS_Term : A Term object representing T_0 in the equation above.

    RHS_Terms : A list of Term objects whose ith entry represents T_i in the
    equation above.

    Device: The device (gpu or cpu) that we train on.

    ----------------------------------------------------------------------------
    Returns:

    A tuple. The first entry of the tuple is a scalar tensor whose lone element
    contains the mean square collocation loss at the Coords. The second is a
    1D tensor whose ith entry holds the PDE residual at the ith collocation
    point. You can safely discard the second return variable if you just want
    to get the loss. """

    # Make sure Xi's length matches RHS_Terms'.
    assert(torch.numel(Xi) == len(RHS_Terms));

    # First, evaluate U at the Coords.
    U_Coords : torch.Tensor = U(Coll_Points).view(-1);



    ############################################################################
    # Form a dictionary housing D_j U, for each derivative D_j in Derivatives.

    # Initialize
    D_U_Dict : Dict[Derivative] = {};

    # Cycle through the derivatives.
    for j in range(len(Derivatives)):
        # Fetch D_j.
        D_j : Derivative = Derivatives[j];

        # Check if we can compute D_j from any of the derivatives of U that
        # we've already computed.
        for i in range(j, 0):
            # Get the ith derivative.
            D_i : Derivative = Derivatives[i];

            # Check if D_i is a child of D_j.
            if(D_i.Is_Child_Of(D_j)):
                # If so, compute D_j U from D_i U.
                D_j_U : torch.Tensor = Derivative_From_Derivative(
                                        Da      = D_j,
                                        Db      = D_i,
                                        Db_U    = D_U_Dict[tuple(D_i.Encoding)],
                                        Coords  = Coords).view(-1);


                # Store the result in the dictionary.
                D_U_Dict[tuple(D_j.Encoding)] = D_j_U;

                # Break!
                break;
        else:
            # This runs if we do not encounter break in the for loop above.
            # If we end up here, then we can not calculate D_j U from a
            # derivative that we already computed. In this case, we must compute
            # D_j U from U_Coords.
            I : Derivative = Derivative(Encoding = numpy.array([0, 0]));

            # Compute D_j U.
            D_j_U : torch.Tensor = Derivative_From_Derivative(
                                    Da      = D_j,
                                    Db      = I,
                                    Db_U    = U_Coords,
                                    Coords  = Coords).view(-1);

            # Store the result in the dictionary.
            D_U_Dict[tuple(D_j.Encoding)] = D_j_U;



    ############################################################################
    # Construct b(U) (see doc string).

    # Initialize b_U to a vector of all ones.
    b_U : torch.Tensor = torch.ones_like(U_Coords);

    # Cycle through the sub-terms of the LHS Term.
    for i in range(LHS_Term.Num_Sub_Terms):
        # First, fetch the sub-term's derivative.
        D_i : Derivative = LHS_Term.Derivatives[i];

        # Next, fetch its value from the dictionary.
        D_i_U : torch.Tensor = D_U_Dict[tuple(D_i.Encoding)];

        # Accumulate D_i_U into b_U.
        b_U = torch.multiply(b_U, D_i_U);



    ############################################################################
    # Construct L(U)*Xi (see doc string).

    L_U_Xi : torch.Tensor = torch.zeros_like(b_U);

    # Cycle through the RHS Terms.
    for j in range(len(RHS_Terms)):
        # Get the term.
        T_j     : Term          = RHS_Terms[j];

        # Initialize a tensor to hold T_j(U).
        T_j_U   : torch.Tesnor  = torch.ones_like(U_Coords);

        # Cycle through T_j's sub-terms.
        for i in range(T_j.Num_Sub_Terms):
            # First, fetch the sub-term's derivative.
            D_i : Derivative = LHS_Term.Derivatives[i];

            # Next, fetch its value from the dictionary.
            D_i_U : torch.Tensor = D_U_Dict[tuple(D_i.Encoding)];

            # Accumulate D_i_U into T_j_U.
            T_j_U = torch.multiply(T_j_U, D_i_U);

        # Accumulate T_j_U*Xi[j] into L_U_Xi.
        L_U_Xi += torch.multiply(T_j_U, Xi[j]);

    # Now, compute the residual, b(U) - L(U)Xi.
    Residual : torch.Tensor = torch.subtract(b_U, L_U_Xi);

    # Return the mean square residual (Collocation Loss), and the residual.
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
