import numpy as np;
import torch;
from typing import Tuple;

from Network import Neural_Network;
from Loss import Data_Loss, Coll_Loss, Lp_Loss, L0_Approx_Loss;
from Mappings import    Col_Number_to_Multi_Index_Class, \
                        Index_to_x_Derivatives, Index_to_xy_Derivatives_Class;



def Training(
        U                           : Neural_Network,
        Xi                          : torch.Tensor,
        Coll_Points                 : torch.Tensor,
        Data_Points                 : torch.Tensor,
        Data_Values                 : torch.Tensor,
        Highest_Order_Derivatives   : int,
        Index_to_Derivatives,
        Col_Number_to_Multi_Index   : Col_Number_to_Multi_Index_Class,
        p                           : float,
        Lambda                      : float,
        Optimizer                   : torch.optim.Optimizer,
        Device                      : torch.device = torch.device('cpu')) -> None:
    """ This function runs one epoch of training. We enforce the learned PDE
    (library-Xi product) at the Coll_Points. We also make U match the
    Data_Values at the Data_Points.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coeffcients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Data_Points: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If U accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Data_Values: A tensor holding the value of the true solution at the data
    points. If Data_Points has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Highest_Order_Derivatives: The highest order spatial partial derivatives of
    U that are present in the library terms.

    Index_to_Derivatives: A mapping which sends sub-index values to spatial
    partial derivatives. This is needed to build the library in Coll_Loss.
    If U is a function of 1 spatial variable, this should be the function
    Index_to_x_Derivatives. If U is a function of two spatial variables, this
    should be an instance of Index_to_xy_Derivatives.

    Col_Number_to_Multi_Index: A mapping which sends column numbers to
    Multi-Indices. Coll_Loss needs this function. This should be an instance of
    the Col_Number_to_Multi_Index_Class class.

    p, Lambda: the settings value for p and Lambda (in the loss function).

    optimizer: the optimizer we use to train U and Xi. It should have
    been initialized with both network's parameters.

    Device: The device for U and Xi.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Put U in training mode.
    U.train();

    # Define closure function (needed for LBFGS)
    def Closure():
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Evaluate the Loss (Note, we enforce a BC of 0)
        Loss = (Coll_Loss(
                    U                           = U,
                    Xi                          = Xi,
                    Coll_Points                 = Coll_Points,
                    Highest_Order_Derivatives   = Highest_Order_Derivatives,
                    Index_to_Derivatives        = Index_to_Derivatives,
                    Col_Number_to_Multi_Index   = Col_Number_to_Multi_Index,
                    Device                      = Device)

                +

                Data_Loss(
                    U           = U,
                    Data_Points = Data_Points,
                    Data_Values = Data_Values)

                +

                Lambda*Lp_Loss( Xi = Xi,
                                p = p));

        # Back-propigate to compute gradients of Loss with respect to network
        # parameters (only do if this if the loss requires grad)
        if (Loss.requires_grad == True):
            Loss.backward();

        return Loss;

    # update network parameters.
    Optimizer.step(Closure);



def Testing(
        U                           : Neural_Network,
        Xi                          :  Neural_Network,
        Coll_Points                 : torch.Tensor,
        Data_Points                 : torch.Tensor,
        Data_Values                 : torch.Tensor,
        Highest_Order_Derivatives   : int,
        Index_to_Derivatives,
        Col_Number_to_Multi_Index   : Col_Number_to_Multi_Index_Class,
        p                           : float,
        Lambda                      : float,
        Device                      : torch.device = torch.device('cpu')) -> Tuple[float, float]:
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

    Data_Points: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Data_Values: A tensor holding the value of the true solution at the data
    points. If Data_Points has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Highest_Order_Derivatives: The highest order spatial partial derivatives of
    U that are present in the library terms.

    Index_to_Derivatives: A mapping which sends sub-index values to spatial
    partial derivatives. This is needed to build the library in Coll_Loss.
    If U is a function of 1 spatial variable, this should be the function
    Index_to_x_Derivatives. If U is a function of two spatial variables, this
    should be an instance of Index_to_xy_Derivatives.

    Col_Number_to_Multi_Index: A mapping which sends column numbers to
    Multi-Indices. Coll_Loss needs this function. This should be an instance of
    the Col_Number_to_Multi_Index_Class class.

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
            Data_Points = Data_Points,
            Data_Values = Data_Values).item();

    Coll_Loss_Value : float = Coll_Loss(
            U                           = U,
            Xi                          = Xi,
            Coll_Points                 = Coll_Points,
            Highest_Order_Derivatives   = Highest_Order_Derivatives,
            Index_to_Derivatives        = Index_to_Derivatives,
            Col_Number_to_Multi_Index   = Col_Number_to_Multi_Index,
            Device                      = Device).item();

    Lambda_Lp_Loss_Value : float = Lambda*L0_Approx_Loss(
            Xi = Xi,
            s = p).item();

    # Return the losses.
    return (Data_Loss_Value, Coll_Loss_Value, Lambda_Lp_Loss_Value);
