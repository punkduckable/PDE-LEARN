# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

import  torch;
from    typing     import List, Tuple, Dict, Callable;

from    Network    import Network;
from    Loss       import Data_Loss, Coll_Loss, Lp_Loss, L2_Squared_Loss;
from    Derivative import Derivative;
from    Term       import Term;


def Training(   U               : List[Network],
                Xi              : torch.Tensor,
                Coll_Points     : List[torch.Tensor],
                Inputs          : List[torch.Tensor],
                Targets         : List[torch.Tensor],
                Derivatives     : List[Derivative],
                LHS_Term        : Term,
                RHS_Terms       : List[Term],
                p               : float,
                Weights         : Dict[str, float],
                Optimizer       : torch.optim.Optimizer,
                Device          : torch.device = torch.device('cpu')) -> Dict:
    """ 
    I need to update this.


    This function runs one epoch of training. We enforce the learned PDE 
    (library-Xi product) at the Coll_Points. We also make U match the Targets at
    the Inputs.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coefficients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If U accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Data-point.

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

    p: the settings value for p in "Lp" loss function.
    
    Weights: A dictionary of floats. It should have keys for "Lp", "Coll", and 
    "Data".

    Optimizer: the optimizer we use to train U and Xi. It should have
    been initialized with both network's parameters.

    Device: The device for U and Xi.

    ----------------------------------------------------------------------------
    Returns:

    A dictionary with the following keys:
        "Coll Loss", "Data Loss", "Lp Loss": floats housing the value of the
        corresponding loss.

        "Total Loss": a float housing the total loss.

        "Residual": A 1D tensor whose ith entry holds the value of the PDE
        residual at the ith collocation point.
    """

    assert(len(U) == len(Coll_Points));
    assert(len(U) == len(Inputs));
    assert(len(U) == len(Targets));

    Num_DataSets : int = len(U);

    # Put each U in training mode.
    for i in range(Num_DataSets):
        U[i].train();

    # Initialize variables to track the residual, losses. We need to do this
    # because we find these variables in the Closure function (which has its own
    # scope. Thus, any variables created in Closure are inaccessible from
    # outside Closure).
    Residual_List       : List[float] = [];
    Coll_Loss_List      : List[float] = [0]*Num_DataSets;
    Data_Loss_List      : List[float] = [0]*Num_DataSets;
    L2_Loss_List        : List[float] = [0]*Num_DataSets;
    Lp_Loss_Buffer      = 0.0;
    Total_Loss_List     : List[float] = [0]*Num_DataSets;

    for i in range(Num_DataSets):
        Residual_List.append(torch.empty(Coll_Points[i].shape[0], dtype = torch.float32));

    # Define closure function (needed for LBFGS)
    def Closure() -> torch.Tensor:
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Set up buffers to hold the losses
        Coll_Loss_Value     = torch.zeros(1, dtype = torch.float32);
        Data_Loss_Value     = torch.zeros(1, dtype = torch.float32);
        L2_Loss_Value       = torch.zeros(1, dtype = torch.float32);
        Total_Loss_Value    = torch.zeros(1, dtype = torch.float32);

        # First, calculate the L2 loss, since it is not specific to each data set.
        Lp_Loss_Value = Lp_Loss(    Xi      = Xi,
                                    p       = p);
        Lp_Loss_Buffer = Lp_Loss_Value.detach().item()

        # Now calculate the losses for each data set.
        for i in range(Num_DataSets):
            # Get the collocation, data, and L2 loss for the ith data set.
            ith_Coll_Loss_Value, ith_Residual = Coll_Loss(
                                            U           = U[i],
                                            Xi          = Xi,
                                            Coll_Points = Coll_Points[i],
                                            Derivatives = Derivatives,
                                            LHS_Term    = LHS_Term,
                                            RHS_Terms   = RHS_Terms,
                                            Device      = Device);

            ith_Data_Loss_Value = Data_Loss(U                   = U[i],
                                            Inputs              = Inputs[i],
                                            Targets             = Targets[i]);

            ith_L2_Loss_Value = L2_Squared_Loss(U = U[i]);

            ith_Total_Loss_Value = (Weights["Data"]*ith_Data_Loss_Value + 
                                    Weights["Coll"]*ith_Coll_Loss_Value + 
                                    Weights["Lp"]*Lp_Loss_Value + 
                                    Weights["L2"]*ith_L2_Loss_Value);

            # Store those losses in the buffers (for the returned dict)
            Residual_List[i][:]   = ith_Residual.detach();
            Coll_Loss_List[i]  = ith_Coll_Loss_Value.detach().item();
            Data_Loss_List[i] = ith_Data_Loss_Value.detach().item();
            L2_Loss_List[i]    = ith_L2_Loss_Value.detach().item();
            Total_Loss_List[i] = ith_Total_Loss_Value.detach().item();

            # Finally, accumulate the losses.
            Coll_Loss_Value     += ith_Coll_Loss_Value;
            Data_Loss_Value     += ith_Data_Loss_Value;
            L2_Loss_Value       += ith_L2_Loss_Value;
            Total_Loss_Value    += ith_Total_Loss_Value;
        
        # Back-propagate to compute gradients of Total_Loss with respect to
        # network parameters (only do if this if the loss requires grad)
        if (Total_Loss_Value.requires_grad == True):
            Total_Loss_Value.backward();

        return Total_Loss_Value;

    # update network parameters.
    Optimizer.step(Closure);

    # Return the residual tensor.
    return {"Residuals"     : Residual_List,
            "Coll Losses"   : Coll_Loss_List,
            "Data Losses"   : Data_Loss_List,
            "Lp Loss"       : Lp_Loss_Buffer,
            "L2 Losses"     : L2_Loss_List,
            "Total Losses"  : Total_Loss_List};



def Testing(    U               : List[Network],
                Xi              : Network,
                Coll_Points     : List[torch.Tensor],
                Inputs          : List[torch.Tensor],
                Targets         : List[torch.Tensor],
                Derivatives     : List[Derivative],
                LHS_Term        : Term,
                RHS_Terms       : List[Term],
                p               : float,
                Weights         : Dict[str, float],
                Device          : torch.device = torch.device('cpu')) -> Dict[str, float]:
    """ 
    UPDATE ME


    This function evaluates the losses.

    Note: You CAN NOT run this function with no_grad set True. Why? Because we
    need to evaluate derivatives of U with respect to its inputs to evaluate
    Coll_Loss! Thus, we need torch to build a computational graph.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coefficients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Data-point.

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

    A dictionary with the following keys:
        "Coll Loss", "Data Loss", "Lp Loss": floats housing the value of the
        corresponding loss.

        "Total Loss": a float housing the total loss. 
    """

    assert(len(U) == len(Coll_Points));
    assert(len(U) == len(Inputs));
    assert(len(U) == len(Targets));
    
    Num_DataSets : int = len(U);

    # Put each U in evaluation mode
    for i in range(Num_DataSets):
        U[i].eval();
    
    # First, evaluate the Lp loss, since this does not depend on the data set.
    Lp_Loss_Value : float = Lp_Loss(    Xi    = Xi,
                                        p     = p).item();

    # Get the losses for each data set.
    Data_Loss_List  : List[float] = [0]*Num_DataSets;
    Coll_Loss_List  : List[float] = [0]*Num_DataSets;
    L2_Loss_List    : List[float] = [0]*Num_DataSets;
    Total_Loss_List : List[float] = [0]*Num_DataSets;

    for i in range(Num_DataSets):
        Data_Loss_List[i] = Data_Loss(  U           = U[i],
                                        Inputs      = Inputs[i],
                                        Targets     = Targets[i]).item();

        Coll_Loss_List[i] = Coll_Loss(  U           = U[i],
                                        Xi          = Xi,
                                        Coll_Points = Coll_Points[i],
                                        Derivatives = Derivatives,
                                        LHS_Term    = LHS_Term,
                                        RHS_Terms   = RHS_Terms,
                                        Device      = Device)[0].item();

        L2_Loss_List[i] = L2_Squared_Loss(U = U[i]).item();

        Total_Loss_List[i] =          ( Weights["Data"]*Data_Loss_List[i] + 
                                        Weights["Coll"]*Coll_Loss_List[i] + 
                                        Weights["Lp"]*Lp_Loss_Value + 
                                        Weights["L2"]*L2_Loss_List[i]);

    # Return the losses.
    return {"Data Losses"     : Data_Loss_List,
            "Coll Losses"     : Coll_Loss_List,
            "Lp Loss"       : Lp_Loss_Value,
            "L2 Losses"       : L2_Loss_List,
            "Total Losses"    : Total_Loss_List};
