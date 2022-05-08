# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

import  numpy as np;
import  torch;
from    typing     import Tuple, List;

from Network    import Neural_Network;
from Loss       import Data_Loss, Coll_Loss, Lp_Loss, L0_Approx_Loss;
from Derivative import Derivative;
from Term       import Term;


def Training(   U               : Neural_Network,
                Xi              : torch.Tensor,
                Coll_Points     : torch.Tensor,
                Inputs          : torch.Tensor,
                Targets         : torch.Tensor,
                Derivatives     : List[Derivative],
                LHS_Term        : Term,
                RHS_Terms       : List[Term],
                p               : float,
                Lambda          : float,
                Optimizer       : torch.optim.Optimizer,
                Device          : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function runs one epoch of training. We enforce the learned PDE
    (library-Xi product) at the Coll_Points. We also make U match the
    Targets at the Inputs.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coeffcients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If U accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Targets: A tensor holding the value of the true solution at the data
    points. If Inputs has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Derivatives: We try to learn a PDE of the form
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

    p, Lambda: the settings value for p and Lambda (in the loss function).

    Optimizer: the optimizer we use to train U and Xi. It should have
    been initialized with both network's parameters.

    Device: The device for U and Xi.

    ----------------------------------------------------------------------------
    Returns:

    a 1D tensor whose ith entry holds the value of the collocation loss at the
    ith collocation point. We use this for adaptive collocation points. You can
    safely discard this argument if you want to. """

    # Put U in training mode.
    U.train();

    # Initialize Residual variable.
    Residual = torch.empty(Coll_Points.size()[0], dtype = torch.float32);

    # Define closure function (needed for LBFGS)
    def Closure():
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Evaluate the Losses
        Coll_Loss_Value, Resid = Coll_Loss( U           = U,
                                            Xi          = Xi,
                                            Coll_Points = Coll_Points,
                                            Derivatives = Derivatives,
                                            LHS_Term    = LHS_Term,
                                            RHS_Terms   = RHS_Terms,
                                            Device      = Device);

        Data_Loss_Value : torch.Tensor  = Data_Loss(
                                            U                   = U,
                                            Inputs              = Inputs,
                                            Targets             = Targets);

        Lp_Loss_Value   : torch.tensor  = Lp_Loss(
                                            Xi      = Xi,
                                            p       = p);

        # Evaluate the loss.
        Loss    : torch.Tensor  = Data_Loss_Value + Coll_Loss_Value + Lambda*Lp_Loss_Value;

        # Assign the residual.
        Residual[:] = Resid;

        # Back-propigate to compute gradients of Loss with respect to network
        # parameters (only do if this if the loss requires grad)
        if (Loss.requires_grad == True):
            Loss.backward();

        return Loss;

    # update network parameters.
    Optimizer.step(Closure);

    # Return the residual tensor.
    return Residual;



def Testing(    U               : Neural_Network,
                Xi              : Neural_Network,
                Coll_Points     : torch.Tensor,
                Inputs          : torch.Tensor,
                Targets         : torch.Tensor,
                Derivatives     : List[Derivative],
                LHS_Term        : Term,
                RHS_Terms       : List[Term],
                p               : float,
                Lambda          : float,
                Device          : torch.device = torch.device('cpu')) -> Tuple[float, float]:
    """ This function evaluates the losses.

    Note: You CAN NOT run this function with no_grad set True. Why? Because we
    need to evaluate derivatives of U with respect to its inputs to evaluate
    Coll_Loss! Thus, we need torch to build a computational graph.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coeffcients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Targets: A tensor holding the value of the true solution at the data
    points. If Targets has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Derivatives: We try to learn a PDE of the form
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

    p, Lambda: the settings value for p and Lambda (in the loss function).

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    a tuple of floats. The first element holds the Data_Loss. The second
    holds the Coll_Loss. The third holds Lambda times the Lp_Loss. """

    # Put U in evaluation mode
    U.eval();

    # Get the losses
    Data_Loss_Value : float  = Data_Loss(
            U           = U,
            Inputs      = Inputs,
            Targets     = Targets).item();

    Coll_Loss_Value : float = Coll_Loss(U           = U,
                                        Xi          = Xi,
                                        Coll_Points = Coll_Points,
                                        Derivatives = Derivatives,
                                        LHS_Term    = LHS_Term,
                                        RHS_Terms   = RHS_Terms,
                                        Device      = Device)[0].item();

    Lambda_Lp_Loss_Value : float = Lambda*Lp_Loss(
                                            Xi    = Xi,
                                            p     = p).item();

    # Return the losses.
    return (Data_Loss_Value, Coll_Loss_Value, Lambda_Lp_Loss_Value);
