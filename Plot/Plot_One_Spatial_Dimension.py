# Nonsense to add Code, Classes directories to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Get the path to the Code, Classes directories.
Code_path       = os.path.join(Main_Path, "Code");

# Add them to the Python search path.
sys.path.append(Code_path);

# Code files.
from Network                import Neural_Network;
from Plot_Settings_Reader   import Settings_Reader, Settings_Container;

import torch;
import numpy;
import scipy.io;
import matplotlib.pyplot as pyplot;




def Plot_U_1D(  Num_Hidden_Layers   : int,
                Units_Per_Layer     : int,
                Activation_Function : str,
                Device              : torch.device,
                Load_File_Name      : str,
                t_Coords_Matrix     : numpy.ndarray,
                x_Coords_Matrix     : numpy.ndarray,
                Inputs              : numpy.ndarray,
                Targets_Matrix      : numpy.ndarray) -> None:
    """ To do :D

    Note: This can only plot networks with 1 spatial variable. """


    ############################################################################
    # Setup.

    # First, set up the network.
    U = Neural_Network(
            Num_Hidden_Layers   = Num_Hidden_Layers,
            Neurons_Per_Layer   = Units_Per_Layer,
            Input_Dim           = 2,
            Output_Dim          = 1,
            Activation_Function = Activation_Function,
            Device              = Device);

    # Next, Load U.
    Load_File_Path : str = "../Saves/" + Load_File_Name;
    Saved_State = torch.load(Load_File_Path, map_location = Device);
    U.load_state_dict(Saved_State["U"]);

    # Evaluate the network at these coordinates.
    U_Coords    : torch.Tensor  = U(torch.from_numpy(Inputs)).view(-1);
    U_matrix    : numpy.ndarray = U_Coords.detach().numpy().reshape(Targets_Matrix.shape);


    ############################################################################
    # Plot the solution.

    # first, set figure size.
    pyplot.figure(figsize = (10, 5));

    # Get bounds.
    epsilon : float = .0001;
    U_min : float = numpy.min(U_matrix) - epsilon;
    U_max : float = numpy.max(U_matrix) + epsilon;

    # Set up Axes object.
    Ax = pyplot.subplot(1, 2, 1);
    Ax.set_aspect('auto', adjustable = 'datalim');
    Ax.set_box_aspect(1.);

    # Plot!
    pyplot.contourf(    t_Coords_Matrix,
                        x_Coords_Matrix,
                        U_matrix,
                        levels      = numpy.linspace(U_min, U_max, 500),
                        cmap        = pyplot.cm.jet);
    pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
    pyplot.xlabel("t");
    pyplot.ylabel("x");
    pyplot.title("Neural Network Approximation");


    ############################################################################
    # Plot the residual.

    # Find Residual.
    Residual_Matrix : numpy.ndarray = numpy.subtract(U_matrix, Targets_Matrix);

    # Get bounds.
    Residual_min : float = numpy.min(Residual_Matrix) - epsilon;
    Residual_max : float = numpy.max(Residual_Matrix) + epsilon;

    # Set up Axes object.
    Ax = pyplot.subplot(1, 2, 2);
    Ax.set_aspect('auto', adjustable = 'datalim');
    Ax.set_box_aspect(1.);

    # Plot!
    pyplot.contourf(    t_Coords_Matrix,
                        x_Coords_Matrix,
                        Residual_Matrix,
                        levels      = numpy.linspace(Residual_min, Residual_max, 500),
                        cmap        = pyplot.cm.jet);
    pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
    pyplot.xlabel("t");
    pyplot.ylabel("x");
    pyplot.title("Residual (Approximate minus true solution)");


    ############################################################################
    # Show the plot.

    pyplot.tight_layout();
    pyplot.show();



if __name__ == "__main__":
    # Get Settings.
    Settings : Settings_Container = Settings_Reader();

    ############################################################################
    # Set up the data.

    # Load data file.
    Data_File_Path : str    = "../MATLAB/Data/" + Settings.Mat_File_Name + ".mat";
    File                    = scipy.io.loadmat(Data_File_Path);

    # Fetch spatial, temporal coordinates and the true solution. We cast these
    # to singles (32 bit fp) since that's what PDE-LEARN uses.
    t_Points        : numpy.ndarray = File['t'].reshape(-1).astype(dtype = numpy.float32);
    x_Points        : numpy.ndarray = File['x'].reshape(-1).astype(dtype = numpy.float32);
    Targets_Matrix  : numpy.ndarray = (numpy.real(File['usol'])).astype( dtype = numpy.float32);

    # Determine problem bounds.
    Input_Bounds : numpy.ndarray    = numpy.empty(shape = (2, 2), dtype = numpy.float32);
    Input_Bounds[0, 0]              = t_Points[ 0];
    Input_Bounds[0, 1]              = t_Points[-1];
    Input_Bounds[1, 0]              = x_Points[ 0];
    Input_Bounds[1, 1]              = x_Points[-1];

    # Generate the grid of (t, x) coordinates. The i,j entry of usol should
    # hold the value of the solution at the i,j coordinate.
    t_Coords_Matrix, x_Coords_Matrix = numpy.meshgrid(t_Points, x_Points);

    # Now, stitch successive the rows of the coordinate matrices together
    # to make a 1D array. We interpert the result as a 1 column matrix.
    t_Coords_1D : numpy.ndarray = t_Coords_Matrix.flatten().reshape(-1, 1);
    x_Coords_1D : numpy.ndarray = x_Coords_Matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    Inputs : numpy.ndarray      = numpy.hstack((t_Coords_1D, x_Coords_1D));


    ############################################################################
    # Plot!

    Plot_U_1D(  Num_Hidden_Layers   = Settings.Num_Hidden_Layers,
                Units_Per_Layer     = Settings.Units_Per_Layer,
                Activation_Function = Settings.Activation_Function,
                Device              = torch.device('cpu'),
                Load_File_Name      = Settings.Load_File_Name,
                t_Coords_Matrix     = t_Coords_Matrix,
                x_Coords_Matrix     = x_Coords_Matrix,
                Inputs              = Inputs,
                Targets_Matrix      = Targets_Matrix);
