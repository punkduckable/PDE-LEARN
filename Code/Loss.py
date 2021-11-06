import numpy;
import torch;

from Network import Neural_Network;

from Mappings import xy_Derivatives_To_Index, Index_to_xy_Derivatives, \
                     Num_Multi_Indices, Multi_Indices_Array, \
                     Multi_Index_To_Col_Number, Col_Number_To_Multi_Index;

def Data_Loss(
        U : Neural_Network,
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
