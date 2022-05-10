# Nonsense to add Code, Classes directories to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Get the path to the Code, Classes directories.
Code_Path       = os.path.join(Main_Path, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add them to the Python search path.
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# Code files.
from Network                import Neural_Network;
from Derivative             import Derivative;
from Term                   import Term;
from Plot_Settings_Reader   import Settings_Reader, Settings_Container;
from Loss                   import Coll_Loss;

import  torch;
import  numpy;
import  scipy.io;
from    typing  import List;
import  matplotlib.pyplot as pyplot;



def Plot_U( Num_Hidden_Layers       : int,
            Units_Per_Layer         : int,
            Activation_Function     : str,
            Device                  : torch.device,
            Load_File_Name          : str,
            Derivatives             : Derivative,
            LHS_Term                : Term,
            RHS_Terms               : List[Term],
            t_Coords_Matrix         : numpy.ndarray,
            x_Coords_Matrix         : numpy.ndarray,
            Inputs                  : numpy.ndarray,
            Targets_Matrix          : numpy.ndarray) -> None:
    """ This function plots U, the error between U and the data set, and the
    PDE Residual. Currently, this function only works if the underlying dataset
    is in MATLAB. Further, this function only works for datasets with one
    spatial dimension.

    ----------------------------------------------------------------------------
    Argumnets:

    Num_Hidden_Layers: The number of hidden layers in the network.

    Units_Per_Layer: The number of hidden units per hidden layer in the network.

    Activation_Function: The activation function of the network.

    Device: The device we want to load the network on.

    Load_File_Name: The Save file that contains the network paramaters + Xi.

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

    t_Coords_Matrix, x_coords_matrix: We evaluate U, the error, and the PDE
    Residual on a grid of coordinates. These arguments are matricies whose i,j
    entry holds the value of the x, t coordinate at the (i,j)th coordinate,
    respectively.

    Inputs: This is a M by 2 tensor (where M is the number of grid points),
    whose kth row holds the coordinates of the kth gird point. This is literally
    just  [ flatten(t_coords_matrix), flatten(x_coords_matrix) ]. I made it a
    separate argument just for convenience.

    Targets_Matrix: This is a matrix with the same shape as t_coords_matrix/
    x_coords_matrix. It's i,j entry holds the target value at the (i,j)th
    coordinate.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """


    ############################################################################
    # Setup.

    # First, initialize U.
    U = Neural_Network(
            Num_Hidden_Layers   = Num_Hidden_Layers,
            Neurons_Per_Layer   = Units_Per_Layer,
            Input_Dim           = 2,
            Output_Dim          = 1,
            Activation_Function = Activation_Function,
            Device              = Device);

    # Next, intiailize Xi.
    Xi = torch.zeros(   len(RHS_Terms),
                        dtype           = torch.float32,
                        device          = Device,
                        requires_grad   = True);

    # Next, Load U, Xi.
    Load_File_Path : str = "../Saves/" + Load_File_Name;
    Saved_State = torch.load(Load_File_Path, map_location = Device);

    U.load_state_dict(Saved_State["U"]);
    Xi = Saved_State["Xi"];

    # Map Inputs to a tensor.
    torch_Inputs : torch.Tensor = torch.from_numpy(Inputs);

    # Evaluate the network at these coordinates.
    U_Coords    : torch.Tensor  = U(torch_Inputs).detach().view(-1);
    U_matrix    : numpy.ndarray = U_Coords.detach().numpy().reshape(Targets_Matrix.shape);

    # Use the Coll_Loss function to obtain the PDE residual at the Inputs. We
    # do this in batches to conserve memory.
    Residual_Coords : torch.Tensor = torch.empty_like(U_Coords);

    Num_Coords  : int = torch_Inputs.shape[0];
    Num_Batches : int = Num_Coords // 1000;
    for i in range(Num_Batches):
        Residual_Coords[i*1000 : (i + 1)*1000] = Coll_Loss(
                                                    U           = U,
                                                    Xi          = Xi,
                                                    Coll_Points = torch_Inputs[i*1000:(i + 1)*1000, :],
                                                    Derivatives = Derivatives,
                                                    LHS_Term    = LHS_Term,
                                                    RHS_Terms   = RHS_Terms,
                                                    Device      = Device)[1].detach();
    # Clean up.
    if(Num_Coords % 1000 != 0):
        Residual_Coords[Num_Batches*1000:] = Coll_Loss(
                                                U           = U,
                                                Xi          = Xi,
                                                Coll_Points = torch_Inputs[Num_Batches*1000:, :],
                                                Derivatives = Derivatives,
                                                LHS_Term    = LHS_Term,
                                                RHS_Terms   = RHS_Terms,
                                                Device      = Device)[1].detach();


    ############################################################################
    # Plot the solution.

    # first, set figure size.
    pyplot.figure(figsize = (15, 4));

    # Get bounds.
    epsilon : float = .0001;
    U_min : float = numpy.min(U_matrix) - epsilon;
    U_max : float = numpy.max(U_matrix) + epsilon;

    # Set up Axes object.
    Ax = pyplot.subplot(1, 3, 1);
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
    # Plot the Error.

    # Find Error.
    Error_Matrix : numpy.ndarray = numpy.subtract(U_matrix, Targets_Matrix);

    # Get bounds.
    Error_min : float = numpy.min(Error_Matrix) - epsilon;
    Error_max : float = numpy.max(Error_Matrix) + epsilon;

    # Set up Axes object.
    Ax = pyplot.subplot(1, 3, 2);
    Ax.set_aspect('auto', adjustable = 'datalim');
    Ax.set_box_aspect(1.);

    # Plot!
    pyplot.contourf(    t_Coords_Matrix,
                        x_Coords_Matrix,
                        Error_Matrix,
                        levels      = numpy.linspace(Error_min, Error_max, 500),
                        cmap        = pyplot.cm.jet);
    pyplot.colorbar(location = "right", fraction = 0.046, pad = 0.04);
    pyplot.xlabel("t");
    pyplot.ylabel("x");
    pyplot.title("Error (Approximate minus true solution)");


    ############################################################################
    # Plot the PDE residual.

    # Find PDE Residual.
    Residual_Matrix : numpy.ndarray = Residual_Coords.detach().numpy().reshape(Targets_Matrix.shape);

    # Get bounds.
    Residual_min : float = numpy.min(Residual_Matrix) - epsilon;
    Residual_max : float = numpy.max(Residual_Matrix) + epsilon;

    # Set up Axes object.
    Ax = pyplot.subplot(1, 3, 3);
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
    pyplot.title("PDE Residual");


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

    Plot_U( Num_Hidden_Layers       = Settings.Num_Hidden_Layers,
            Units_Per_Layer         = Settings.Units_Per_Layer,
            Activation_Function     = Settings.Activation_Function,
            Device                  = torch.device('cpu'),
            Load_File_Name          = Settings.Load_File_Name,
            Derivatives             = Settings.Derivatives,
            LHS_Term                = Settings.LHS_Term,
            RHS_Terms               = Settings.RHS_Terms,
            t_Coords_Matrix         = t_Coords_Matrix,
            x_Coords_Matrix         = x_Coords_Matrix,
            Inputs                  = Inputs,
            Targets_Matrix          = Targets_Matrix);
