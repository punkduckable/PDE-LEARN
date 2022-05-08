# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Readers_Path    = os.path.join(Code_Path, "Readers");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);
sys.path.append(Classes_Path);

import  numpy;
import  torch;
import  time;
from    typing import List;

from Settings_Reader    import Settings_Reader, Settings_Container;
from Data_Loader        import Data_Loader;
from Derivative         import Derivative;
from Term               import Term;
from Network            import Rational, Neural_Network;
from Test_Train         import Testing, Training;
from Loss               import Data_Loss, Lp_Loss, Coll_Loss;
from Points             import Generate_Points;
from Xi                 import Print_PDE;



def main():
    # Load the settings, print them.
    Settings = Settings_Reader();
    for (Setting, Value) in Settings.__dict__.items():
        print(("%-25s = " % Setting) + str(Value));

    # Start a setup timer.
    Setup_Timer : float = time.perf_counter();
    print("Setting up... ", end = '');


    ############################################################################
    # Set up Data
    # This sets up the testing/training inputs/targets, and the input bounds.
    # This will also tell us the number of spatial dimensions (there's a row of
    # Input_Bounds for each coordinate component. Since one coordinate is for
    # time, one minus the number of rows gives the number of spatial dimensions).

    Data_Container = Data_Loader(   DataSet_Name = Settings.DataSet_Name,
                                    Device       = Settings.Device);

    # Get the number of input dimensions.
    Settings.Num_Spatial_Dimensions : int = Data_Container.Input_Bounds.shape[0] - 1;

    # Now, determine how many library terms we have. This determines Xi's size.
    Num_Library_Terms : int = len(Settings.RHS_Terms)


    ############################################################################
    # Set up U and Xi.

    U = Neural_Network(
            Num_Hidden_Layers   = Settings.Num_Hidden_Layers,
            Neurons_Per_Layer   = Settings.Units_Per_Layer,
            Input_Dim           = Settings.Num_Spatial_Dimensions + 1,
            Output_Dim          = 1,
            Activation_Function = Settings.Activation_Function,
            Device              = Settings.Device);

    # We set up Xi as a Parameter for.... complicated reasons. In pytorch, a
    # paramater is basically a special tensor that is supposed to be a trainable
    # part of a module. It acts just like a regular tensor, but almost always
    # has requires_grad set to true. Further, since it's a sub-class of Tensor,
    # we can distinguish it from regular Tensors. In particular, optimizers
    # expect a list or dictionary of Parameters... not Tensors. Since we want
    # to train Xi, we set it up as a Parameter.
    Xi = torch.zeros(   Num_Library_Terms,
                        dtype           = torch.float32,
                        device          = Settings.Device,
                        requires_grad   = True);


    ############################################################################
    # Load U, Xi

    # First, check if we should load Xi, U from file. If so, load them!
    if( Settings.Load_U         == True or
        Settings.Load_Xi        == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        if(Settings.Load_U == True):
            U.load_state_dict(Saved_State["U"]);

        if(Settings.Load_Xi == True):
            Xi = Saved_State["Xi"];


    ############################################################################
    # Set up the optimizer.
    # Note: we need to do this after loading Xi, since loading Xi potentially
    # overwrites the original Xi (loading the optimizer later ensures the
    # optimizer optimizes the correct Xi tensor).

    Params = list(U.parameters());
    Params.append(Xi);

    if  (Settings.Optimizer == "Adam"):
        Optimizer = torch.optim.Adam(Params, lr = Settings.Learning_Rate);
    elif(Settings.Optimizer == "LBFGS"):
        Optimizer = torch.optim.LBFGS(Params, lr = Settings.Learning_Rate);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Settings.Optimizer));
        exit();


    if(Settings.Load_Optimizer  == True ):
        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        # Now load the optimizer.
        Optimizer.load_state_dict(Saved_State["Optimizer"]);

        # Enforce the new learning rate (do not use the saved one).
        for param_group in Optimizer.param_groups:
            param_group['lr'] = Settings.Learning_Rate;


    # Setup is now complete. Report time.
    print("Done! Took %7.2fs" % (time.perf_counter() - Setup_Timer));


    ############################################################################
    # Run the Epochs!

    # Set up targeted collocation points.
    Targeted_Coll_Pts = torch.empty((0, Settings.Num_Spatial_Dimensions + 1), dtype = torch.float32);

    # Set up timer.
    Epoch_Timer : float = time.perf_counter();

    # Epochs!!!
    print("Running %d epochs..." % Settings.Num_Epochs);
    for t in range(1, Settings.Num_Epochs + 1):
        # First, generate new training collocation points.
        Random_Coll_Points = Generate_Points(
                                Bounds      = Data_Container.Input_Bounds,
                                Num_Points  = Settings.Num_Train_Coll_Points,
                                Device      = Settings.Device);

        # Now, append the targeted collocation points from the last epoch.
        Train_Coll_Points : torch.Tensor = torch.vstack((Random_Coll_Points, Targeted_Coll_Pts));

        # Now run a Training Epoch. Keep track of the residual.
        Residual = Training(    U           = U,
                                Xi          = Xi,
                                Coll_Points = Train_Coll_Points,
                                Inputs      = Data_Container.Train_Inputs,
                                Targets     = Data_Container.Train_Targets,
                                Derivatives = Settings.Derivatives,
                                LHS_Term    = Settings.LHS_Term,
                                RHS_Terms   = Settings.RHS_Terms,
                                p           = Settings.p,
                                Lambda      = Settings.Lambda,
                                Optimizer   = Optimizer,
                                Device      = Settings.Device);

        # Find the Absolute value of the residuals. Isolate those corresponding
        # to the "random" collocation points.
        Abs_Residual        : torch.Tensor = torch.abs(Residual);
        Random_Residuals    : torch.Tensor = Abs_Residual[:Settings.Num_Train_Coll_Points];

        # Evaluate the mean, standard deviation of the absolute residual at the
        # random points.
        Residual_Mean   : torch.Tensor = torch.mean(Random_Residuals);
        Residual_SD     : torch.Tensor = torch.std(Random_Residuals);

        # Determine which collocation points have residuals that are more than
        # 2 STD above the mean.
        Cutoff                  : float         = Residual_Mean + 3*Residual_SD
        Big_Residual_Indices    : torch.Tensor  = torch.greater_equal(Abs_Residual, Cutoff);

        # Keep the corresponding collocation points.
        Targeted_Coll_Pts       : torch.Tensor  = Train_Coll_Points[Big_Residual_Indices, :].detach();

        # Test the code (and print the loss) every 10 Epochs. For all other
        # epochs, print the Epoch to indicate the program is making progress.
        if(t % 10 == 0 or t == 1):
            # Generate new testing Collocation Coordinates
            Test_Coll_Points = Generate_Points(
                            Bounds      = Data_Container.Input_Bounds,
                            Num_Points  = Settings.Num_Test_Coll_Points,
                            Device      = Settings.Device);

            # Evaluate losses on training points.
            (Train_Data_Loss, Train_Coll_Loss, Train_Lp_Loss) = Testing(
                U           = U,
                Xi          = Xi,
                Coll_Points = Train_Coll_Points,
                Inputs      = Data_Container.Train_Inputs,
                Targets     = Data_Container.Train_Targets,
                Derivatives = Settings.Derivatives,
                LHS_Term    = Settings.LHS_Term,
                RHS_Terms   = Settings.RHS_Terms,
                p           = Settings.p,
                Lambda      = Settings.Lambda,
                Device      = Settings.Device);

            # Evaluate losses on the testing points.
            (Test_Data_Loss, Test_Coll_Loss, Test_Lp_Loss) = Testing(
                U           = U,
                Xi          = Xi,
                Coll_Points = Train_Coll_Points,
                Inputs      = Data_Container.Test_Inputs,
                Targets     = Data_Container.Test_Targets,
                Derivatives = Settings.Derivatives,
                LHS_Term    = Settings.LHS_Term,
                RHS_Terms   = Settings.RHS_Terms,
                p           = Settings.p,
                Lambda      = Settings.Lambda,
                Device      = Settings.Device);

            # Print losses!
            print("Epoch #%-4d | Test: \t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (t, Test_Data_Loss, Test_Coll_Loss, Test_Lp_Loss, Test_Data_Loss + Test_Coll_Loss + Test_Lp_Loss));
            print("            | Train:\t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (Train_Data_Loss, Train_Coll_Loss, Train_Lp_Loss, Train_Data_Loss + Train_Coll_Loss + Train_Lp_Loss));
        else:
            print("Epoch #%-4d | Targeted %3d \t Cutoff = %g"   % (t, Targeted_Coll_Pts.shape[0], Cutoff));

    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs," % Epoch_Runtime);
    print("an average of %7.2fs per epoch." % (Epoch_Runtime / Settings.Num_Epochs));


    ############################################################################
    # Threshold Xi.

    # Cycle through components of Xi. Remove all whose magnitude is smaller
    # than the threshold.
    Pruned_Xi = torch.empty_like(Xi);
    N   : int = Xi.numel();
    for k in range(N):
        Abs_Xi_k = abs(Xi[k].item());
        if(Abs_Xi_k < Settings.Threshold):
            Pruned_Xi[k] = 0;
        else:
            Pruned_Xi[k] = Xi[k];


    ############################################################################
    # Report final PDE

    Print_PDE(  Xi          = Pruned_Xi,
                LHS_Term    = Settings.LHS_Term,
                RHS_Terms   = Settings.RHS_Terms);


    ############################################################################
    # Save.

    if(Settings.Save_State == True):
        Save_File_Path : str = "../Saves/" + Settings.Save_File_Name;
        torch.save({"U"         : U.state_dict(),
                    "Xi"        : Xi,
                    "Optimizer" : Optimizer.state_dict()},
                    Save_File_Path);



if(__name__ == "__main__"):
    main();
