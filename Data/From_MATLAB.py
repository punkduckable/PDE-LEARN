
import numpy;
import random;
import scipy.io;
import matplotlib.pyplot as pyplot;

from Create_Data_Set import Create_Data_Set;



Make_Plot : bool = True;

def main():
    # Specify settings.
    Data_File_Name          : str   = "Allen_Cahn";
    Num_Spatial_Dimensions  : int   = 1;
    Noise_Proportion        : float = .75;

    Num_Train_Examples      : int   = 5000;
    Num_Test_Examples       : int   = 1000;

    # Now pass them to "From_MATLAB".
    if(Num_Spatial_Dimensions == 1):
        From_MATLAB_1D( Data_File_Name      = Data_File_Name,
                        Noise_Proportion    = Noise_Proportion,
                        Num_Train_Examples  = Num_Train_Examples,
                        Num_Test_Examples   = Num_Test_Examples);
    else: # Num_Spatial_Dimensions == 2
        From_MATLAB_2D( Data_File_Name      = Data_File_Name,
                        Noise_Proportion    = Noise_Proportion,
                        Num_Train_Examples  = Num_Train_Examples,
                        Num_Test_Examples   = Num_Test_Examples);




def From_MATLAB_1D( Data_File_Name      : str,
                    Noise_Proportion    : float,
                    Num_Train_Examples  : int,
                    Num_Test_Examples   : int) -> None:
    """ This function loads a .mat data set with 1D data (one spatial variable,
    and one temporal variable) and generates a sparse and noisy data set from
    it. To do this, we first read in a .mat data set. We assume this file
    contains three fields: t, x, and usol. t and x are ordered lists of the t
    and x grid lines (lines along which there are gridpoints), respectively. We
    assume that the values in x are uniformly spaced. u sol contains the value
    of the true solution at each gridpoint. Each row of usol contains the
    solution for a particular position, while each column contains the solution
    for a particular time.

    We then add the desired noise level (Noise_Proportion*100% noise) to usol,
    yielding a noisy data set. Next, we draw a sample of Num_Train_Examples
    from the set of coordinates, along with the corresponding elements of noisy
    data set. This becomes our Training data set. We draw another sample of
    Num_Test_Examples from the set of coordinates along with the
    corresponding elements of the noisy data set. These become our Testing set.

    ----------------------------------------------------------------------------
    Arguments:

    Data_File_Name: A string containing the name of a .mat file (without the
    extension) that houses the matlab data set we want to read.

    Noise_Proportion: The noise level we want to introduce.

    Num_Train_Examples, Num_Test_Examples: The number of Training/Testing
    examples we want, respectively.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Load data file.
    Data_File_Path = "../MATLAB/Data/" + Data_File_Name + ".mat";
    data_in        = scipy.io.loadmat(Data_File_Path);

    # Fetch spatial, temporal coordinates and the true solution. We cast these
    # to singles (32 bit fp) since that's what PDE-LEARN uses.
    t_points : numpy.ndarray = data_in['t'].reshape(-1).astype(dtype = numpy.float32);
    x_points : numpy.ndarray = data_in['x'].reshape(-1).astype(dtype = numpy.float32);
    Data_Set : numpy.ndarray = (numpy.real(data_in['usol'])).astype( dtype = numpy.float32);

    # Determine problem bounds.
    Input_Bounds : numpy.ndarray    = numpy.empty(shape = (2, 2), dtype = numpy.float32);
    Input_Bounds[0, 0]              = t_points[ 0];
    Input_Bounds[0, 1]              = t_points[-1];
    Input_Bounds[1, 0]              = x_points[ 0];
    Input_Bounds[1, 1]              = x_points[-1];

    # Add noise to true solution.
    Noisy_Data_Set = Data_Set + (Noise_Proportion)*numpy.std(Data_Set)*numpy.random.randn(*Data_Set.shape);

    # Generate the grid of (t, x) coordinates. The i,j entry of usol should
    # hold the value of the solution at the i,j coordinate.
    t_coords_matrix, x_coords_matrix = numpy.meshgrid(t_points, x_points);

    if(Make_Plot == True):
        epsilon : float = .0001;
        Data_min : float = numpy.min(Noisy_Data_Set) - epsilon;
        Data_max : float = numpy.max(Noisy_Data_Set) + epsilon;

        # Plot!
        pyplot.contourf(    t_coords_matrix,
                            x_coords_matrix,
                            Noisy_Data_Set,
                            levels      = numpy.linspace(Data_min, Data_max, 500),
                            cmap        = pyplot.cm.jet);

        pyplot.colorbar();
        pyplot.xlabel("t");
        pyplot.ylabel("x");
        pyplot.show();

    # Now, stitch successive the rows of the coordinate matrices together
    # to make a 1D array. We interpert the result as a 1 column matrix.
    t_coords_1D : numpy.ndarray = t_coords_matrix.flatten().reshape(-1, 1);
    x_coords_1D : numpy.ndarray = x_coords_matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    All_Data_Coords : numpy.ndarray = numpy.hstack((t_coords_1D, x_coords_1D));
    All_Data_Values : numpy.ndarray = Noisy_Data_Set.flatten();

    # Next, generate the Testing/Training sets. To do this, we sample a uniform
    # distribution over subsets of {1, ... , N} of size Num_Train_Examples,
    # and another over subsets of {1, ... , N} of size Num_Test_Examples.
    # Here, N is the number of coordinates.
    Train_Indicies : numpy.ndarray = numpy.random.choice(All_Data_Coords.shape[0], Num_Train_Examples, replace = False);
    Test_Indicies  : numpy.ndarray = numpy.random.choice(All_Data_Coords.shape[0], Num_Test_Examples , replace = False);

    # Now select the corresponding testing, training data points/values.
    Train_Inputs    = All_Data_Coords[Train_Indicies, :];
    Train_Targets   = All_Data_Values[Train_Indicies];

    Test_Inputs     = All_Data_Coords[Test_Indicies, :];
    Test_Targets    = All_Data_Values[Test_Indicies];

    # Send everything to Create_Data_Set
    DataSet_Name : str = (  Data_File_Name + "_" +
                            "N" + str(int(100*Noise_Proportion)) + "_" +
                            "P" + str(Num_Train_Examples) );

    Create_Data_Set(    Name            = DataSet_Name,
                        Train_Inputs    = Train_Inputs,
                        Train_Targets   = Train_Targets,
                        Test_Inputs     = Test_Inputs,
                        Test_Targets    = Test_Targets,
                        Input_Bounds    = Input_Bounds);



def From_MATLAB_2D( Data_File_Name      : str,
                    Noise_Proportion    : float,
                    Num_Train_Examples  : int,
                    Num_Test_Examples   : int) -> None:
    """ This function loads a .mat data set with 2D data (two spatial variables,
    and one temporal variable) and generates a sparse and noisy data set from
    it. To do this, we first read in a .mat data set. We assume this file
    contains four fields: t, x, y, and usol. t, x, and y are ordered lists of
    the t, x, and y t grid lines (lines along which there are gridpoints),
    respectively. We assume that the values in x and y are uniformly spaced.
    usol is a gigantic 3d array whose i, j, k entry holds the value of the
    solution at (t_i, x_j, y_k), where t_i is the ith entry of t, x_j is the
    jth entry of x, and y_k is the kth entry of y.

    We then add the desired noise level (Noise_Proportion*100% noise) to usol,
    yielding a noisy data set. Next, we draw a sample of Num_Train_Examples
    from the set of coordinates, along with the corresponding elements of noisy
    data set. This becomes our Training data set. We draw another sample of
    Num_Test_Examples from the set of coordinates along with the
    corresponding elements of the noisy data set. These become our Testing set.

    ----------------------------------------------------------------------------
    Arguments:

    Data_File_Name: A string containing the name of a .mat file (without the
    extension) that houses the matlab data set we want to read.

    Noise_Proportion: The noise level we want to introduce.

    Num_Train_Examples, Num_Test_Examples: The number of Training/Testing
    examples we want, respectively.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Load data file.
    Data_File_Path = "../MATLAB/Data/" + Data_File_Name + ".mat";
    data_in        = scipy.io.loadmat(Data_File_Path);

    # Fetch spatial, temporal coordinates and the true solution. We cast these
    # to singles (32 bit fp) since that's what PDE-LEARN uses.
    t_points : numpy.ndarray = data_in['t'].reshape(-1).astype(dtype = numpy.float32);
    x_points : numpy.ndarray = data_in['x'].reshape(-1).astype(dtype = numpy.float32);
    y_points : numpy.ndarray = data_in['y'].reshape(-1).astype(dtype = numpy.float32);
    Data_Set : numpy.ndarray = (numpy.real(data_in['usol'])).astype( dtype = numpy.float32);

    # Determine problem bounds.
    Input_Bounds : numpy.ndarray    = numpy.empty(shape = (3, 2), dtype = numpy.float32);
    Input_Bounds[0, 0]              = t_points[ 0];
    Input_Bounds[0, 1]              = t_points[-1];
    Input_Bounds[1, 0]              = x_points[ 0];
    Input_Bounds[1, 1]              = x_points[-1];
    Input_Bounds[2, 0]              = y_points[ 0];
    Input_Bounds[2, 1]              = y_points[-1];

    # Add noise to true solution.
    Noisy_Data_Set : numpy.ndarray = Data_Set + (Noise_Proportion)*numpy.std(Data_Set)*numpy.random.randn(*Data_Set.shape);

    # Generate the grid of (t, x, y) coordinates. The i,j,k entry of usol should
    # hold the value of the solution at the i,j,k coordinate.
    t_coords_matrix, x_coords_matrix, y_coords_matrix = numpy.meshgrid(t_points, x_points, y_points);

    # Now, stitch successive the rows of the coordinate matricies together
    # to make a 1d array. We interpert the result as a 1 column matrix.
    t_coords_1D : numpy.ndarray = t_coords_matrix.flatten().reshape(-1, 1);
    x_coords_1D : numpy.ndarray = x_coords_matrix.flatten().reshape(-1, 1);
    y_coords_1D : numpy.ndarray = y_coords_matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    All_Data_Coords : numpy.ndarray = numpy.hstack((t_coords_1D, x_coords_1D, y_coords_1D));
    All_Data_Values : numpy.ndarray = Noisy_Data_Set.flatten();

    # Next, generate the Testing/Training sets. To do this, we sample a uniform
    # distribution over subsets of {1, ... , N} of size Num_Train_Examples,
    # and another over subsets of {1, ... , N} of size Num_Test_Examples.
    # Here, N is the number of coordinates.
    Train_Indicies : numpy.ndarray = numpy.random.choice(All_Data_Coords.shape[0], Num_Train_Examples, replace = False);
    Test_Indicies  : numpy.ndarray = numpy.random.choice(All_Data_Coords.shape[0], Num_Test_Examples , replace = False);

    # Now select the corresponding testing, training data points/values.
    Train_Inputs    = All_Data_Coords[Train_Indicies, :];
    Train_Targets   = All_Data_Values[Train_Indicies];

    Test_Inputs     = All_Data_Coords[Test_Indicies, :];
    Test_Targets    = All_Data_Values[Test_Indicies];

    # Send everything to Create_Data_Set
    DataSet_Name : str = (  Data_File_Name + "_" +
                            "N" + str(int(100*Noise_Proportion)) + "_" +
                            "P" + str(Num_Train_Examples) );

    Create_Data_Set(    Name            = DataSet_Name,
                        Train_Inputs    = Train_Inputs,
                        Train_Targets   = Train_Targets,
                        Test_Inputs     = Test_Inputs,
                        Test_Targets    = Test_Targets,
                        Input_Bounds    = Input_Bounds);

if __name__ == "__main__":
    main();
