# Nonsense to add Code, Classes directories to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path       = os.path.dirname(os.path.abspath(os.path.curdir));

# Get the path to the Code, Classes directories.
Code_Path       = os.path.join(Main_Path, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add them to the Python search path.
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# Code files.
from    Network                 import  Network;
from    Derivative              import  Derivative;
from    Term                    import  Term, Build_Term_From_State;
from    Plot_Settings_Reader    import  Settings_Reader;
from    Loss                    import  Coll_Loss;
from    typing                  import  Dict, List;

import  torch;
import  numpy;
import  scipy.io;
from    typing                  import  List;
import  matplotlib.pyplot       as      pyplot;



def Plot_U( Load_File_Name          : str,
            Mat_File_Names          : List[str],
            Device                  : torch.device,
            t_Coords_Matrix_List    : List[numpy.ndarray],
            x_Coords_Matrix_List    : List[numpy.ndarray],
            Targets_Matrix_List     : List[numpy.ndarray]) -> None:
    """ This function plots U, the error between U and the data set, and the
    PDE Residual for each data set in the saved file. Currently, this function 
    only works if the underlying dataset comes from MATLAB. Further, this 
    function only works for datasets with one spatial dimension.

    ----------------------------------------------------------------------------
    Arguments:

    Load_File_Name: The Save file that contains each solution network's saved
    state, the library, and Xi.

    Mat_File_Names: A list housing the names of the mat files used for each 
    data set. We use this to name the figures.

    Device: The device we want to load the network on.

    t_Coords_Matrix_List, x_coords_matrix_List: These are lists of numpy 
    ndarray objects. We evaluate the kth solution network, U[k], the its error 
    with the kth data set, and its PDE residual on a grid of coordinates. 
    The kth entry of t/x_Coords_Matrix_List is a matrix whose i,j entry holds 
    the value of the t and x coordinate at the (i,j)th coordinate, 
    respectively.

    Targets_Matrix_List: This is list of numpy ndarray objects. Its kth entry
    is a matrix with the same shape as t_Coords_Matrix_List[k] or 
    x_Coords_Matrix_List[k]. The i,j entry of the kth list item holds the 
    target value at the (i,j)th coordinate of the kth data set.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """


    ############################################################################
    # Setup.

    # First, load the file.     
    Load_File_Path      : str       = "../Saves/" + Load_File_Name;
    Saved_State         : Dict      = torch.load(Load_File_Path, map_location = Device);

    # Determine the number of data sets. This MUST match the length of
    # coords and targets list arguments.
    Num_DataSets        : int       = len(Saved_State["U States"]);
    assert(Num_DataSets == len(x_Coords_Matrix_List));
    assert(Num_DataSets == len(t_Coords_Matrix_List));
    assert(Num_DataSets == len(Targets_Matrix_List));

    # Set up each U.
    U_List : List[Network] = [];
    for k in range(Num_DataSets):
        Uk_State            : Dict      = Saved_State["U States"][k];
        Widths              : List[int] = Uk_State["Widths"];
        Hidden_Activation   : str       = Uk_State["Activation Types"][0];
        Output_Activation   : str       = Uk_State["Activation Types"][-1];

        # Initialize U.
        U_List.append(Network(   Widths              = Widths, 
                            Hidden_Activation   = Hidden_Activation, 
                            Output_Activation   = Output_Activation,
                            Device              = Device));

        # Finally, load U[k]'s state.
        U_List[k].Set_State(Uk_State);

    # Next, load Xi.
    Xi : torch.Tensor = Saved_State["Xi"];

    # Next, load the derivatives
    Derivatives     : List[Derivative]  = [];
    Num_Derivatives : int               = len(Saved_State["Derivative Encodings"]);
    for i in range(Num_Derivatives):
        Derivatives.append(Derivative(Encoding = Saved_State["Derivative Encodings"][i]));
        
    # Finally, load the library (LHS term and RHS Terms)
    LHS_Term        : Term          = Build_Term_From_State(State = Saved_State["LHS Term State"]);
    RHS_Terms       : List[Term]    = [];

    Num_RHS_Terms   : int           = len(Saved_State["RHS Term States"]);
    for i in range(Num_RHS_Terms):
        RHS_Terms.append(Build_Term_From_State(State = Saved_State["RHS Term States"][i]));


    ############################################################################
    # Evaluate each U, PDE residual on inputs.

    Error_Matrix_List       : List[numpy.ndarray] = [];
    U_Matrix_List           : List[numpy.ndarray] = [];
    Residual_Matrix_List    : List[numpy.ndarray] = [];
    U_Matrix_List           : List[numpy.ndarray] = [];

    for k in range(Num_DataSets):
        # First, stitch successive the rows of the kth coordinate matrices 
        # together to make a 1D array. We interpret the result as a 1 column
        # matrix.
        kth_t_Coords_1D : numpy.ndarray = t_Coords_Matrix_List[k].flatten().reshape(-1, 1);
        kth_x_Coords_1D : numpy.ndarray = x_Coords_Matrix_List[k].flatten().reshape(-1, 1);

        # Generate a matrix of coordinates (one coordinate per row).
        kth_Inputs          : numpy.ndarray = numpy.hstack((kth_t_Coords_1D, kth_x_Coords_1D));

        # Map Inputs to a tensor.
        kth_torch_Inputs    : torch.Tensor  = torch.from_numpy(kth_Inputs);

        # Evaluate the network at these coordinates.
        kth_U_Coords        : torch.Tensor  = U_List[k](kth_torch_Inputs).detach().view(-1);
        U_Matrix_List.append(kth_U_Coords.detach().numpy().reshape(Targets_Matrix_List[k].shape));

        # Evaluate the kth error
        Error_Matrix_List.append(numpy.subtract(U_Matrix_List[k], Targets_Matrix_List[k]));

        # Use the Coll_Loss function to obtain the PDE residual at the Inputs. We
        # do this in batches to conserve memory.
        kth_Residual_Coords : torch.Tensor  = torch.empty_like(kth_U_Coords);

        Num_Coords      : int           = kth_torch_Inputs.shape[0];
        Num_Batches     : int           = Num_Coords // 1000;
        for i in range(Num_Batches):
            kth_Residual_Coords[i*1000 : (i + 1)*1000] = Coll_Loss(
                                                        U           = U_List[k],
                                                        Xi          = Xi,
                                                        Coll_Points = kth_torch_Inputs[i*1000:(i + 1)*1000, :],
                                                        Derivatives = Derivatives,
                                                        LHS_Term    = LHS_Term,
                                                        RHS_Terms   = RHS_Terms,
                                                        Device      = Device)[1].detach();
        # Clean up.
        if(Num_Coords % 1000 != 0):
            kth_Residual_Coords[Num_Batches*1000:] = Coll_Loss(
                                                    U           = U_List[k],
                                                    Xi          = Xi,
                                                    Coll_Points = kth_torch_Inputs[Num_Batches*1000:, :],
                                                    Derivatives = Derivatives,
                                                    LHS_Term    = LHS_Term,
                                                    RHS_Terms   = RHS_Terms,
                                                    Device      = Device)[1].detach();
        kth_Residual_Matrix = kth_Residual_Coords.detach().numpy().reshape(Targets_Matrix_List[k].shape)
        Residual_Matrix_List.append(kth_Residual_Matrix);


    ############################################################################
    # Plot

    # First, set up a folder to save the plots in.
    Plot_Directory_Name : str = "Plots_" + Load_File_Name;
    Plot_Directory_Path : str = "../Figures/"   + Plot_Directory_Name;
    os.mkdir(Plot_Directory_Path);

    Figure_Index : int = 0;
    for k in range(Num_DataSets):
        ############################################################################
        # Plot the solution.

        # Get bounds.
        epsilon : float = .0001;
        kth_U_min   : float = numpy.min(U_Matrix_List[k]) - epsilon;
        kth_U_max   : float = numpy.max(U_Matrix_List[k]) + epsilon;

        # Plot!
        pyplot.figure(Figure_Index);
        pyplot.clf();
        pyplot.contourf(    t_Coords_Matrix_List[k],
                            x_Coords_Matrix_List[k],
                            U_Matrix_List[k],
                            levels      = numpy.linspace(kth_U_min, kth_U_max, 500),
                            cmap        = pyplot.cm.jet);
        pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
        pyplot.xlabel("t");
        pyplot.ylabel("x");
        pyplot.title("Neural Network Approximation");
        pyplot.savefig(Plot_Directory_Path + "/U_" + Mat_File_Names[k] + ".png", dpi = 500);

        Figure_Index += 1;

        ############################################################################
        # Plot the Error.

        # Get bounds.
        kth_Error_min : float = numpy.min(Error_Matrix_List[k]) - epsilon;
        kth_Error_max : float = numpy.max(Error_Matrix_List[k]) + epsilon;

        # Plot!
        pyplot.figure(Figure_Index);
        pyplot.clf();
        pyplot.contourf(    t_Coords_Matrix_List[k],
                            x_Coords_Matrix_List[k],
                            Error_Matrix_List[k],
                            levels      = numpy.linspace(kth_Error_min, kth_Error_max, 500),
                            cmap        = pyplot.cm.jet);
        pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
        pyplot.xlabel("t");
        pyplot.ylabel("x");
        pyplot.title("Error (Approximate minus true solution)");
        pyplot.savefig(Plot_Directory_Path + "/Error_" + Mat_File_Names[k] + ".png", dpi = 500);

        Figure_Index += 1;

        ############################################################################
        # Plot the PDE residual.

        # Get bounds.
        kth_Residual_min : float = numpy.min(Residual_Matrix_List[k]) - epsilon;
        kth_Residual_max : float = numpy.max(Residual_Matrix_List[k]) + epsilon;

        # Plot!
        pyplot.figure(Figure_Index);
        pyplot.clf();
        pyplot.contourf(    t_Coords_Matrix_List[k],
                            x_Coords_Matrix_List[k],
                            Residual_Matrix_List[k],
                            levels      = numpy.linspace(kth_Residual_min, kth_Residual_max, 500),
                            cmap        = pyplot.cm.jet);
        pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
        pyplot.xlabel("t");
        pyplot.ylabel("x");
        pyplot.title("PDE Residual");
        pyplot.savefig(Plot_Directory_Path + "/PDE_Residual_" + Mat_File_Names[k] + ".png", dpi = 500);

        Figure_Index += 1;
    

    ############################################################################
    # Show the plot.

    pyplot.show();



if __name__ == "__main__":
    Settings : Dict = Settings_Reader();


    ############################################################################
    # Set up the data.

    Num_DataSets : int = len(Settings["Mat File Names"]);

    # Load each data set.
    t_Coords_Matrix_List    : List[numpy.ndarray] = [];
    x_Coords_Matrix_List    : List[numpy.ndarray] = [];
    Targets_Matrix_List     : List[numpy.ndarray] = [];

    for k in range(Num_DataSets):
        # Load data file.
        kth_Data_File_Path : str    = "../MATLAB/Data/" + Settings["Mat File Names"][k] + ".mat";
        kth_File                    = scipy.io.loadmat(kth_Data_File_Path);

        # Fetch spatial, temporal coordinates and the true solution. We cast these
        # to singles (32 bit fp) since that's what PDE-LEARN uses.
        t_Points            : numpy.ndarray = kth_File['t'].reshape(-1).astype(dtype = numpy.float32);
        x_Points            : numpy.ndarray = kth_File['x'].reshape(-1).astype(dtype = numpy.float32);

        kth_Targets_Matrix  : numpy.ndarray = (numpy.real(kth_File['usol'])).astype( dtype = numpy.float32);
        Targets_Matrix_List.append(kth_Targets_Matrix);

        # Generate the grid of (t, x) coordinates. The i,j entry of usol should
        # hold the value of the solution at the i,j coordinate.
        kth_t_Coords_Matrix, kth_x_Coords_Matrix = numpy.meshgrid(t_Points, x_Points);
        t_Coords_Matrix_List.append(kth_t_Coords_Matrix);
        x_Coords_Matrix_List.append(kth_x_Coords_Matrix);


    ############################################################################
    # Plot!

    Plot_U( Load_File_Name          = Settings["Load File Name"],
            Mat_File_Names          = Settings["Mat File Names"],
            Device                  = torch.device('cpu'),
            t_Coords_Matrix_List    = t_Coords_Matrix_List,
            x_Coords_Matrix_List    = x_Coords_Matrix_List,
            Targets_Matrix_List     = Targets_Matrix_List);
