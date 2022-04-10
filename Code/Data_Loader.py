import numpy;
import torch;
import random;
import scipy.io;

from Settings_Reader import Settings_Container;



class Data_Container:
    pass;



def Data_Loader(DataSet_Name   : str,
                Device         : torch.device):
    """ This function loads a DataSet from file, converts it contents to a torch
    Tensor, and returns the result.

    ----------------------------------------------------------------------------
    Arguments:

    DataSet_Name : The name of a file in Data/DataSets (without the .npz
    extension). We load the DataSet in this file.

    Device : The device we're running training on.

    ----------------------------------------------------------------------------
    Returns:

    A Data Container object. What's in that container depends on which mode
    we're in. """

    # Load the DataSet.
    DataSet_Path    = "../Data/DataSets/" + DataSet_Name + ".npz";
    DataSet         = numpy.load(DataSet_Path);

    # Make the Container.
    Container = Data_Container();

    # First, fetch the training/testing inputs and targets.
    Train_Inputs    : numpy.ndarray = DataSet["Train_Inputs"];
    Train_Targets   : numpy.ndarray = DataSet["Train_Targets"];

    Test_Inputs     : numpy.ndarray = DataSet["Test_Inputs"];
    Test_Targets    : numpy.ndarray = DataSet["Test_Targets"];

    # Convert these to tensors and add them to the container.
    Container.Train_Inputs  = torch.from_numpy(Train_Inputs).to(device = Device);
    Container.Train_Targets = torch.from_numpy(Train_Targets).to(device = Device);

    Container.Test_Inputs  = torch.from_numpy(Test_Inputs).to(device = Device);
    Container.Test_Targets = torch.from_numpy(Test_Targets).to(device = Device);

    # Finally, fetch the Input Bounds array.
    Container.Input_Bounds = DataSet["Input_Bounds"];

    # The container is now full. Return it!
    return Container;
