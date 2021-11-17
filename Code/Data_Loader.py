import numpy as np;
import torch;
import random;
import scipy.io;

from Settings_Reader import Settings_Container;



class Data_Container:
    pass;



def Data_Loader(Settings : Settings_Container):
    """ This function loads data from file and returns it. We assume that the
    data is in a .mat file.

    If U is a function of one spatial variable, then the data file must contain
    three fileds: t, x, and usol. t and x are ordered lists which specify the
    positions of the t and x gridlines, respectively. usol is a gigantic matrix
    whose i, j entry holds the solution at (x_i, t_j), where x_i is the ith
    element of x and t_j is the jth entry of t.

    if U is a function of two spatial variables, then the data file must contain
    four fields: t, x, y, and usol. t, x, and y are ordered lists which specify
    the positions of the t, x, and y girdlines, respectively. usol is a a
    gigantic 3d array whose i, j, k entry holds the value of the solution at
    (t_i, x_j, y_k), where t_i is the ith entry of t, x_j is the jth entry of x,
    and y_k is the kth entry of y.

    ----------------------------------------------------------------------------
    Arguments:

    Settings: The Settings_Container object returned by Settings_Reader.

    ----------------------------------------------------------------------------
    Returns:

    A Data Container object. The data_container contains Bounds, testing/
    training coordinates, and testing/training values (the values of the
    noisy solution at the testing/training coordinates). """

    # Load data file.
    Data_File_Path = "../Data/" + Settings.Data_File_Name;
    data_in = scipy.io.loadmat(Data_File_Path);

    # Fetch the true solution.
    U_Sol_In = (np.real(data_in['usol'])).astype(dtype = numpy.float32);

    # Add noise.
    Noisy_Data = U_Sol_In + (Settings.Noise_Proportion)*np.std(U_Sol_In)*np.random.randn(*U_Sol_In.shape);

    # We have to handle the cases "U is a function of 1 spatial variable" and
    # "U is a function of two spatial variables" separatly.
    if(Settings.Num_Spatial_Dimensions == 1):
        # Load t, x values
        t_values = data_in['t'].flatten().astype(dtype = numpy.float32);
        x_values = data_in['x'].flatten().astype(dtype = numpy.float32);

        # Generate the grid of (t, x) coordinates. The i,j entry of usol should
        # hold the value of the solution at the i,j coordinate.
        num_t_values : int = t_values.size;
        num_x_values : int = x_values.size;

        t_coords_matrix = numpy.empty((num_x_values, num_t_values), dtype = numpy.float32);
        x_coords_matrix = numpy.empty((num_x_values, num_t_values), dtype = numpy.float32);

        for i in range(num_x_values):
            for j in range(num_t_values):
                t_coords_matrix[i, j] = t_coords[j];
                x_coords_matrix[i, j] = x_coords[i];

        # Now, stitch successive the rows of the coordinate matricies together
        # to make a 1d array. We interpert the result as a 1 column matrix.
        t_coords_list = t_coords_matrix.reshape(-1, 1);
        x_coords_list = x_coords_matrix.reshape(-1, 1);

        # Stitch successive rows of Noisy_Data together to make a 1d array.
        Noisy_Data_List = Noisy_Data.flatten();

        # horizontally stack the coordinate lists to make a list of coordiinates
        Coords = numpy.hstack((t_coords_list, x_coords_list))

        # Determine the upper and lower spatial/temporal bounds. Technically
        # this won't include part of the domain if we use peroidic BCs, but I'm
        # okay with that.
        x_lower = x_values[ 0];
        x_upper = x_values[-1];
        t_lower = t_values[ 0];
        t_upper = t_values[-1];

        # Initialize data container object.
        Container = Data_Container();
        Container.Bounds           = numpy.array(  ((t_lower, t_upper),
                                                    (x_lower, x_upper)), dtype = numpy.float32);

        # Now, set up the testing, training coordinates. Randomly select
        # Num_Training_Points, Num_Testing_Points coordinate indicies.
        Train_Indicies = np.random.choice(Coords.shape[0], Settings.Num_Train_Data_Points, replace = False);
        Test_Indicies  = np.random.choice(Coords.shape[0], Settings.Num_Test_Data_Points , replace = False);

        # Now select the corresponding testing, training data points, values.
        # Add everything to the Container.
        Container.Train_Coords = torch.from_numpy(Coords[Train_Indicies, :]).to(dtype = torch.float32, device = Settings.Device);
        Container.Train_Data   = torch.from_numpy(Noisy_Data_List[Train_Indicies]).to(dtype = torch.float32, device = Settings.Device);

        Container.Test_Coords = torch.from_numpy(Coords[Test_Indicies, :]).to(dtype = torch.float32, device = Settings.Device);
        Container.Test_Data   = torch.from_numpy(Noisy_Data_List[Test_Indicies]).to(dtype = torch.float32, device = Settings.Device);

        # The container is now full. Return it!
        return Container;

    if(Settings.Num_Spatial_Dimensions == 1):
        # Load t, x, y values
        t_values = data_in['t'].flatten().astype(dtype = numpy.float32);
        x_values = data_in['x'].flatten().astype(dtype = numpy.float32);
        y_values = data_in['y'].flatten().astype(dtype = numpy.float32);

        # Generate the grid of (t, x) coordinates. The i,j entry of usol should
        # hold the value of the solution at the i,j coordinate.
        num_t_values : int = t_values.size;
        num_x_values : int = x_values.size;
        num_y_values : int = y_values.size;

        t_coords_matrix = numpy.empty((num_t_values, num_x_values, num_y_values), dtype = numpy.float32);
        x_coords_matrix = numpy.empty((num_t_values, num_x_values, num_y_values), dtype = numpy.float32);
        x_coords_matrix = numpy.empty((num_t_values, num_x_values, num_y_values), dtype = numpy.float32);

        for i in range(num_t_values):
            for j in range(num_x_values):
                for k in range(num_y_values);
                    t_coords_matrix[i, j, k] = t_coords[i];
                    x_coords_matrix[i, j, k] = x_coords[j];
                    x_coords_matrix[i, j, k] = y_coords[k];


        # Now, stitch successive the rows of the coordinate matricies together
        # to make a 1d array. We interpert the result as a 1 column matrix.
        t_coords_list = t_coords_matrix.flatten().reshape(-1, 1);
        x_coords_list = x_coords_matrix.flatten().reshape(-1, 1);
        y_coords_list = y_coords_matrix.flatten().reshape(-1, 1);

        # Stitch successive rows of Noisy_Data together to make a 1d array.
        Noisy_Data_List = Noisy_Data.flatten();

        # horizontally stack the coordinate lists to make a list of coordiinates
        Coords = numpy.hstack((t_coords_list, x_coords_list, y_coords_list))

        # Determine the upper and lower spatial/temporal bounds. Technically
        # this won't include part of the domain if we use peroidic BCs, but I'm
        # okay with that.
        t_lower = t_values[ 0];
        t_upper = t_values[-1];
        x_lower = x_values[ 0];
        x_upper = x_values[-1];
        y_lower = y_values[ 0];
        y_upper = y_values[-1];

        # Initialize data container object.
        Container = Data_Container();
        Container.Bounds           = numpy.array(  ((t_lower, t_upper),
                                                    (x_lower, x_upper),
                                                    (y_lower, y_upper)), dtype = numpy.float32);

        # Now, set up the testing, training coordinates. Randomly select
        # Num_Training_Points, Num_Testing_Points coordinate indicies.
        Train_Indicies = np.random.choice(Coords.shape[0], Settings.Num_Train_Data_Points, replace = False);
        Test_Indicies  = np.random.choice(Coords.shape[0], Settings.Num_Test_Data_Points , replace = False);

        # Now select the corresponding testing, training data points, values.
        # Add everything to the Container.
        Container.Train_Coords = torch.from_numpy(Coords[Train_Indicies, :]).to(dtype = torch.float32, device = Settings.Device);
        Container.Train_Data   = torch.from_numpy(Noisy_Data_List[Train_Indicies]).to(dtype = torch.float32, device = Settings.Device);

        Container.Test_Coords = torch.from_numpy(Coords[Test_Indicies, :]).to(dtype = torch.float32, device = Settings.Device);
        Container.Test_Data   = torch.from_numpy(Noisy_Data_List[Test_Indicies]).to(dtype = torch.float32, device = Settings.Device);

        # The container is now full. Return it!
        return Container;
