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
from Network                import Network;
from Derivative             import Derivative;
from Term                   import Term, Build_Term_From_State;
from Plot_Settings_Reader   import Settings_Reader;
from Loss                   import Coll_Loss;
from typing                 import Dict, List;

import  torch;
import  numpy;
import  scipy.io;
from    typing  import List;
import  matplotlib.pyplot as pyplot;



def Plot_U( Load_File_Name          : str,
            Device                  : torch.device,
            t_Coords_Matrix         : numpy.ndarray,
            x_Coords_Matrix         : numpy.ndarray,
            Inputs                  : numpy.ndarray,
            Targets_Matrix          : numpy.ndarray) -> None:
    """ This function plots U, the error between U and the data set, and the
    PDE Residual. Currently, this function only works if the underlying dataset
    is in MATLAB. Further, this function only works for datasets with one
    spatial dimension.

    ----------------------------------------------------------------------------
    Arguments:

    Load_File_Name: The Save file that contains the network parameters + Xi.

    Device: The device we want to load the network on.

    t_Coords_Matrix, x_coords_matrix: We evaluate U, the error, and the PDE
    Residual on a grid of coordinates. These arguments are matrices whose i,j
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

    # First, load the file. 
    Load_File_Path  : str   = "../Saves/" + Load_File_Name;
    Saved_State     : Dict  = torch.load(Load_File_Path, map_location = Device);

    # Now, set up U.
    U_State             : Dict      = Saved_State["U State"];
    Widths              : List[int] = U_State["Widths"];
    Hidden_Activation   : str       = U_State["Activation Types"][0];
    Output_Activation   : str       = U_State["Activation Types"][-1];

    # Initialize U.
    U = Network(    Widths              = Widths, 
                    Hidden_Activation   = Hidden_Activation, 
                    Output_Activation   = Output_Activation,
                    Device              = Device);

    # Finally, load U's state.
    U.Set_State(Saved_State["U State"]);

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
    # Evaluate U, PDE residual on inputs.

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
    Settings : Dict = Settings_Reader();


    ############################################################################
    # Set up the data.

    # Load data file.
    Data_File_Path : str    = "../MATLAB/Data/" + Settings["Mat File Name"] + ".mat";
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
    # to make a 1D array. We interpret the result as a 1 column matrix.
    t_Coords_1D : numpy.ndarray = t_Coords_Matrix.flatten().reshape(-1, 1);
    x_Coords_1D : numpy.ndarray = x_Coords_Matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    Inputs : numpy.ndarray      = numpy.hstack((t_Coords_1D, x_Coords_1D));


    ############################################################################
    # Plot!

    Plot_U( Load_File_Name          = Settings["Load File Name"],
            Device                  = torch.device('cpu'),
            t_Coords_Matrix         = t_Coords_Matrix,
            x_Coords_Matrix         = x_Coords_Matrix,
            Inputs                  = Inputs,
            Targets_Matrix          = Targets_Matrix);
