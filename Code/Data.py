import  numpy;
import  torch;
from    typing  import List, Dict, Callable;


def Data_Loader(DataSet_Name   : str,
                Device         : torch.device):
    """ 
    This function loads a DataSet from file, converts it contents to a torch
    Tensor, and returns the result.

    ----------------------------------------------------------------------------
    Arguments:

    DataSet_Name : The name of a file in Data/DataSets (without the .npz
    extension). We load the DataSet in this file.

    Device : The device we're running training on.

    ----------------------------------------------------------------------------
    Returns:

    A dictionary with six items. Below is a list of the keys (in quotes) as
    well as their corresponding values:
        "Train Inputs", "Test Inputs": 2D arrays whose ith row holds the
        coordinates of the ith training or testing example, respectively.

        "Test Target", "Train Target": 1 column arrays whose ith entry holds the
        value of the system response function at the ith training and testing
        points, respectively.

        "Input Bounds": A 2 column array whose ith row holds the lower and
        upper bounds of the problem domain along the ith axis.

        "Number Spatial Dimensions": An integer specifying the number of spatial
        dimensions (one less than the number of rows in "Input Bounds").  
    """

    # Load the DataSet.
    DataSet_Path        = "../Data/DataSets/" + DataSet_Name + ".npz";
    DataSet             = numpy.load(DataSet_Path);

    # Now build the return dictionary
    Data_Dict   : Dict  = { "Train Inputs"              : torch.from_numpy(DataSet["Train_Inputs"]).to( device = Device),
                            "Train Targets"             : torch.from_numpy(DataSet["Train_Targets"]).to(device = Device),
                            "Test Inputs"               : torch.from_numpy(DataSet["Test_Inputs"]).to(  device = Device),
                            "Test Targets"              : torch.from_numpy(DataSet["Test_Targets"]).to( device = Device),
                            "Input Bounds"              : DataSet["Input_Bounds"],
                            "Number Spatial Dimensions" : DataSet["Input_Bounds"].shape[0] - 1};

    # All done... return!
    return Data_Dict;
