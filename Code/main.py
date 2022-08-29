# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       : str = os.path.dirname(os.path.abspath(__file__));
Readers_Path    : str = os.path.join(Code_Path, "Readers");
Classes_Path    : str = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);
sys.path.append(Classes_Path);

import  numpy;
import  torch;
import  time;
from    typing import List, Dict, Callable;

from Settings_Reader    import Settings_Reader;
from Data_Loader        import Data_Loader;
from Derivative         import Derivative;
from Term               import Term;
from Network            import Rational, Neural_Network;
from Test_Train         import Testing, Training;
from Loss               import Data_Loss, Lp_Loss, Coll_Loss;
from Points             import Generate_Points;



def main():
    # Load the settings, print them.
    Settings : Dict = Settings_Reader();
    for (Setting, Value) in Settings.items():
        print("%-25s = " % Setting, end = '');

        # Check if Value is a List. If so, print its contents one at a time.
        if(isinstance(Value, list)):
            for i in range(len(Value)):
                print(Value[i], end = '');
                if(i != len(Value) - 1):
                    print(", ", end = '');
                else:
                    print();

        # Otherwise, print the value.
        else:
            print(str(Value));

    # Start a setup timer.
    Setup_Timer : float = time.perf_counter();
    print("Setting up... ", end = '');


    ############################################################################
    # Set up Data
    # This sets up the testing/training inputs/targets, and the input bounds.
    # This will also tell us the number of spatial dimensions (there's a row of
    # Input_Bounds for each coordinate component. Since one coordinate is for
    # time, one minus the number of rows gives the number of spatial dimensions).

    Data_Dict : Dict = Data_Loader( DataSet_Name    = Settings["DataSet Name"],
                                    Device          = Settings["Device"]);

    # Get the number of input dimensions.
    Settings["Num Spatial Dimensions"] : int        = Data_Dict["Number Spatial Dimensions"]

    # Now, determine how many library terms we have. This determines Xi's size.
    Num_Library_Terms : int = len(Settings["RHS Terms"])


    ############################################################################
    # Set up U and Xi.

    U = Neural_Network(
            Num_Hidden_Layers   = Settings["Num Hidden Layers"],
            Neurons_Per_Layer   = Settings["Units Per Layer"],
            Input_Dim           = Settings["Num Spatial Dimensions"] + 1,
            Output_Dim          = 1,
            Activation_Function = Settings["Activation Function"],
            Device              = Settings["Device"]);

    # We set up Xi as a Parameter for.... complicated reasons. In pytorch, a
    # paramater is basically a special tensor that is supposed to be a trainable
    # part of a module. It acts just like a regular tensor, but almost always
    # has requires_grad set to true. Further, since it's a sub-class of Tensor,
    # we can distinguish it from regular Tensors. In particular, optimizers
    # expect a list or dictionary of Parameters... not Tensors. Since we want
    # to train Xi, we set it up as a Parameter.
    Xi = torch.zeros(   Num_Library_Terms,
                        dtype           = torch.float32,
                        device          = Settings["Device"],
                        requires_grad   = True);


    ############################################################################
    # Load U, Xi

    # First, check if we should load Xi, U from file. If so, load them!
    if( Settings["Load U"]      == True or
        Settings["Load Xi"]     == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings["Load File Name"];
        Saved_State = torch.load(Load_File_Path, map_location = Settings["Device"]);

        if(Settings["Load U"] == True):
            U.load_state_dict(Saved_State["U"]);

        if(Settings["Load Xi"] == True):
            Xi = Saved_State["Xi"];


    ############################################################################
    # Set up the optimizer.
    # Note: we need to do this after loading Xi, since loading Xi potentially
    # overwrites the original Xi (loading the optimizer later ensures the
    # optimizer optimizes the correct Xi tensor).

    Params = list(U.parameters());
    Params.append(Xi);

    if  (Settings["Optimizer"] == "Adam"):
        Optimizer = torch.optim.Adam( Params,   lr = Settings["Learning Rate"]);
    elif(Settings["Optimizer"] == "LBFGS"):
        Optimizer = torch.optim.LBFGS(Params,   lr = Settings["Learning Rate"]);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Settings["Optimizer"]));
        exit();


    if(Settings["Load Optimizer"]  == True ):
        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings["Load File Name"];
        Saved_State = torch.load(Load_File_Path, map_location = Settings["Device"]);

        # Now load the optimizer.
        Optimizer.load_state_dict(Saved_State["Optimizer"]);

        # Enforce the new learning rate (do not use the saved one).
        for param_group in Optimizer.param_groups:
            param_group['lr'] = Settings["Learning Rate"];

    # Setup is now complete. Report time.
    print("Done! Took %7.2fs" % (time.perf_counter() - Setup_Timer));


    ############################################################################
    # Run the Epochs!

    # Set up an array to hold the collocation points.
    Targeted_Coll_Pts : torch.Tensor = torch.empty((0, Settings["Num Spatial Dimensions"] + 1), dtype = torch.float32);

    # Set up buffers to hold losses, also set up a timer.
    Epoch_Timer     : float                     = time.perf_counter();
    Train_Losses    : Dict[str, numpy.ndarray]  = {"Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Lp Losses"      : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)};

    Test_Losses    : Dict[str, numpy.ndarray]   = {"Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Lp Losses"      : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)};

    # Epochs!!!
    print("Running %d epochs..." % Settings["Num Epochs"]);
    for t in range(0, Settings["Num Epochs"]):
        ########################################################################
        # Train

        # First, we need to set up the collocation points for this epoch. This
        # set is a combination of randomly generated points and the targeted
        # points from the last epoch.
        Random_Coll_Points  : torch.Tensor = Generate_Points(
                                                Bounds      = Data_Dict["Input Bounds"],
                                                Num_Points  = Settings["Num Train Coll Points"],
                                                Device      = Settings["Device"]);

        Coll_Points         : torch.Tensor = torch.vstack((Random_Coll_Points, Targeted_Coll_Pts));

        # Now run a Training Epoch.
        Train_Dict = Training(  U           = U,
                                Xi          = Xi,
                                Coll_Points = Coll_Points,
                                Inputs      = Data_Dict["Train Inputs"],
                                Targets     = Data_Dict["Train Targets"],
                                Derivatives = Settings["Derivatives"],
                                LHS_Term    = Settings["LHS Term"],
                                RHS_Terms   = Settings["RHS Terms"],
                                p           = Settings["p"],
                                Lambda      = Settings["Lambda"],
                                Optimizer   = Optimizer,
                                Device      = Settings["Device"]);

        # Update the loss history buffers.
        Train_Losses["Data Losses"][t]  = Train_Dict["Data Loss"];
        Train_Losses["Coll Losses"][t]  = Train_Dict["Coll Loss"];
        Train_Losses["Lp Losses"][t]    = Train_Dict["Lp Loss"];
        Train_Losses["Total Losses"][t] = Train_Dict["Total Loss"];


        ########################################################################
        # Test

        # First, generate random collocation points then evaluate the
        # network on them.
        Test_Coll_Points = Generate_Points(
                            Bounds      = Data_Dict["Input Bounds"],
                            Num_Points  = Settings["Num Test Coll Points"],
                            Device      = Settings["Device"]);

        # Evaluate losses on the testing points.
        Test_Dict = Testing(    U           = U,
                                Xi          = Xi,
                                Coll_Points = Test_Coll_Points,
                                Inputs      = Data_Dict["Test Inputs"],
                                Targets     = Data_Dict["Test Targets"],
                                Derivatives = Settings["Derivatives"],
                                LHS_Term    = Settings["LHS Term"],
                                RHS_Terms   = Settings["RHS Terms"],
                                p           = Settings["p"],
                                Lambda      = Settings["Lambda"],
                                Device      = Settings["Device"]);

        Test_Losses["Data Losses"][t]   = Test_Dict["Data Loss"];
        Test_Losses["Coll Losses"][t]   = Test_Dict["Coll Loss"];
        Test_Losses["Lp Losses"][t]     = Test_Dict["Lp Loss"];
        Test_Losses["Total Losses"][t]  = Test_Dict["Total Loss"];


        ########################################################################
        # Update targeted residual points.

        # Find the Absolute value of the residuals. Isolate those corresponding
        # to the "random" collocation points.
        Abs_Residual        : torch.Tensor = torch.abs(Train_Dict["Residual"]);
        Random_Residuals    : torch.Tensor = Abs_Residual[:Settings["Num Train Coll Points"]];

        # Evaluate the mean, standard deviation of the absolute residual at the
        # random points.
        Residual_Mean   : torch.Tensor = torch.mean(Random_Residuals);
        Residual_SD     : torch.Tensor = torch.std(Random_Residuals);

        # Determine which collocation points have residuals that are very far
        # from the mean. At these points, the PDE has a lot of trouble learning
        # something meaningful. The network/PDE needs to adjust its behavior
        # here, so we should hold onto that point.
        Cutoff                  : float         = Residual_Mean + 3*Residual_SD
        Big_Residual_Indices    : torch.Tensor  = torch.greater_equal(Abs_Residual, Cutoff);

        # Keep the corresponding collocation points.
        Targeted_Coll_Pts       : torch.Tensor  = Coll_Points[Big_Residual_Indices, :].detach();


        ########################################################################
        # Report!

        if(t % 10 == 0 or t == Settings["Num Epochs"] - 1):
            # Note: When we print this, the testing/training Lp losses tend to
            # be different. This is because the Training loss is evaluated
            # before backprop, while the Testing loss is evaluated after it (and
            # Xi has been updated).
            print("            | Train:\t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (Train_Dict["Data Loss"], Train_Dict["Coll Loss"], Train_Dict["Lp Loss"], Train_Dict["Total Loss"]));
            print("Epoch #%-4d | Test: \t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (t, Test_Dict["Data Loss"], Test_Dict["Coll Loss"], Test_Dict["Lp Loss"], Test_Dict["Total Loss"]));
        else:
            print("Epoch #%-4d | \t\t Targeted %3d \t\t Cutoff = %g"   % (t, Targeted_Coll_Pts.shape[0], Cutoff));

    # Report runtime!
    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs," % Epoch_Runtime);
    print("an average of %7.2fs per epoch." % (Epoch_Runtime / Settings["Num Epochs"]));


    ############################################################################
    # Threshold Xi.

    # Recall how we enforce Xi: We trick torch into minimizing
    #       [Xi]_1^p + ... + [Xi]_n^p,
    # which is highly concave (for p < 1), by instead minimizing
    #       w_1[Xi]_1^2 + ... + w_n[Xi]_n^2,
    # where each w_i is updated each step to be w_i = [Xi]_i^{2 - p}. The above
    # is convex (if we treat w_1, ... , w_n as constants). There is, however, a
    # problem. If [Xi]_i is smaller than about 3e-4, then [Xi]_i^2 is roughly
    # machine epilon, meaning we run into problems. To aboid this, we instead
    # define
    #       w_i = max{1e-7, [Xi]_i^{p - 2}}.
    # The issue with this approach is that the Lp loss can't really resolve
    # components of Xi which are smaller than about 3e-4. To deal with this, we
    # ignore all components smaller than 5e-4.
    #
    # Note: Switching to double precision would allow us to drop this number
    # further.
    Pruned_Xi = torch.empty_like(Xi);
    N   : int = Xi.numel();
    for k in range(N):
        Abs_Xi_k = abs(Xi[k].item());
        if(Abs_Xi_k < 5e-4):
            Pruned_Xi[k] = 0;
        else:
            Pruned_Xi[k] = Xi[k];


    ############################################################################
    # Report final PDE

    # Print the LHS Term.
    print(Settings["LHS Term"], end = '');
    print(" = ");

    # Print the RHS terms
    for i in range(len(Settings["RHS Terms"])):
        if(Pruned_Xi[i] != 0):
            if(i != 0):
                print(" + ", end = '');
            print("%7.4f" % Pruned_Xi[i], end = '');
            print(Settings["RHS Terms"][i], end = '');

    # End the line.
    print();


    ############################################################################
    # Save.

    # First, come up with a save name that does not conflict with an existing
    # save name. To do this, we first attempt to make a save file name that
    # consists of the data set name plus the activation function and optimizer
    # we used. If a save with that name already exists, we append a "1" to the
    # end of the file name. If that also corresponds to an existing save, then
    # we replace the "1" with a "2" and so on until we get save name that does
    # not already exist.
    Base_File_Name : str = Settings["DataSet Name"] + "_" + Settings["Activation Function"] + "_" + Settings["Optimizer"]

    Counter         : int = 0;
    Save_File_Name  : str = Base_File_Name;
    while(os.path.isfile("../Saves/" + Save_File_Name)):
        # Increment the counter, try appending that onto Base_File_Name.
        Counter         += 1;
        Save_File_Name   = Base_File_Name + ("_%u" % Counter);

    # We can now save!
    torch.save({"U"         : U.state_dict(),
                "Xi"        : Xi,
                "Optimizer" : Optimizer.state_dict()},
                "../Saves/" + Save_File_Name);



if(__name__ == "__main__"):
    main();
