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
from    typing import List, Dict;

from Settings_Reader    import Settings_Reader;
from Library_Reader     import Read_Library;
from Data               import Data_Loader;
from Derivative         import Derivative;
from Term               import Term, Build_Term_From_State;
from Network            import Network;
from Test_Train         import Testing, Training;
from Points             import Generate_Points;
from Plot               import Plot_Losses;



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
    print("\nSetting up...\n");


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


    ############################################################################
    # Set up U, Xi, Library

    # First, if we are loading anything, load in the save.
    if( Settings["Load U"]              == True or 
        Settings["Load Xi, Library"]    == True or
        Settings["Load Optimizer"]      == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings["Load File Name"];
        Saved_State = torch.load(Load_File_Path, map_location = Settings["Device"]);


    # First, either build U or load its state from save. 
    if(Settings["Load U"] == True):
        # First, fetch Widths and Activation functions from U's State.
        U_State             : Dict      = Saved_State["U State"];
        Widths              : List[int] = U_State["Widths"];
        Hidden_Activation   : str       = U_State["Activation Types"][0];
        Output_Activation   : str       = U_State["Activation Types"][-1];

        Settings["Hidden Activation Function"]  = Hidden_Activation;

        # Initialize U.
        U = Network(    Widths              = Widths, 
                        Hidden_Activation   = Hidden_Activation, 
                        Output_Activation   = Output_Activation,
                        Device              = Settings["Device"]);
    
        # Finally, load U's state.
        U.Set_State(Saved_State["U State"]);

        print("Loaded U from state.  ");

    else:
        # First, set up Widths. This is an array whose ith entry specifies the width of 
        # the ith layer of the network (including the input and output layers).
        Widths = [Settings["Num Spatial Dimensions"] + 1] + Settings["Hidden Layer Widths"] + [1];
        
        # Now initialize U.
        U = Network(    Widths              = Widths,
                        Hidden_Activation   = Settings["Hidden Activation Function"],
                        Device              = Settings["Device"]);
        
        print("Set up U using settings in Settings.txt.")

    # Report!
    print("    Hidden Activation: %s" % Settings["Hidden Activation Function"]);
    print("    Widths:            ", end = '');
    for i in range(len(Widths)):
        print(str(Widths[i]), end = '');

        if(i != len(Widths) - 1):  
            print(", ", end = '');
        else:
            print("\n");


    # Second, either build Xi + the library or load it from save.
    if(Settings["Load Xi, Library"] == True):
        # First, load Xi.
        Xi : torch.Tensor = Saved_State["Xi"];

        # Next, load the derivatives
        Derivatives     : List[Derivative]  = [];
        Num_Derivatives : int               = len(Saved_State["Derivative Encodings"]);
        for i in range(Num_Derivatives):
            Derivatives.append(Derivative(Encoding = Saved_State["Derivative Encodings"][i]));
        
        Settings["Derivatives"] = Derivatives;
        
        # Finally, load the library (LHS term and RHS Terms)
        Settings["LHS Term"] = Build_Term_From_State(State = Saved_State["LHS Term State"]);

        RHS_Terms       : List[Term]    = [];
        Num_RHS_Terms   : int           = len(Saved_State["RHS Term States"]);
        for i in range(Num_RHS_Terms):
            RHS_Terms.append(Build_Term_From_State(State = Saved_State["RHS Term States"][i]));
        
        Settings["RHS Terms"] = RHS_Terms;

        print("Loaded Xi, Library from file.");

    else:
        # First, read the library.
        Derivatives, LHS_Term, RHS_Terms    = Read_Library(Settings["Library Path"]);
        Settings["Derivatives"]             = Derivatives;
        Settings["LHS Term"]                = LHS_Term;
        Settings["RHS Terms"]               = RHS_Terms;

        # Next, determine how many library terms we have. This determines Xi's 
        # size.
        Num_RHS_Terms : int = len(Settings["RHS Terms"]);

        # Since we want to learn Xi, we set its requires_Grad to true.
        Xi = torch.zeros(   Num_RHS_Terms,
                            dtype           = torch.float32,
                            device          = Settings["Device"],
                            requires_grad   = True);
                    
        print("Build Xi, Library using settings in Settings.txt");
    
    # Report!
    print("    LHS Term:              %s" % str(Settings["LHS Term"]))
    print("    RHS Terms (%3u total): " % Num_RHS_Terms, end = '');

    for i in range(Num_RHS_Terms):
        print(str(Settings["RHS Terms"][i]), end = '');
        
        if(i != Num_RHS_Terms - 1):
            print(", ", end = '');
        else:
            print("\n");


    ############################################################################
    # Set up the optimizer.
    # Note: we need to do this after loading Xi, since loading Xi potentially
    # overwrites the original Xi (loading the optimizer later ensures the
    # optimizer optimizes the correct Xi tensor).

    Params = list(U.parameters());
    Params.append(Xi);

    if(  Settings["Optimizer"] == "Adam"):
        Optimizer = torch.optim.Adam( Params,   lr = Settings["Learning Rate"]);
    elif(Settings["Optimizer"] == "LBFGS"):
        Optimizer = torch.optim.LBFGS(Params,   lr = Settings["Learning Rate"]);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Settings["Optimizer"]));
        exit();


    if(Settings["Load Optimizer"]  == True ):
        # Now load the optimizer.
        Optimizer.load_state_dict(Saved_State["Optimizer"]);

        # Enforce the new learning rate (do not use the saved one).
        for param_group in Optimizer.param_groups:
            param_group['lr'] = Settings["Learning Rate"];

    # Setup is now complete. Report time.
    print("Set up complete! Took %7.2fs" % (time.perf_counter() - Setup_Timer));


    ############################################################################
    # Run the Epochs!

    # Set up an array to hold the collocation points.
    Targeted_Coll_Pts : torch.Tensor = torch.empty((0, Settings["Num Spatial Dimensions"] + 1), dtype = torch.float32);

    # Set up buffers to hold losses, also set up a timer.
    Epoch_Timer     : float                     = time.perf_counter();
    Train_Losses    : Dict[str, numpy.ndarray]  = {"Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)};

    Test_Losses    : Dict[str, numpy.ndarray]   = {"Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)};

    Parameter_Losses : Dict[str, numpy.ndarray] = {"Lp Losses"      : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                                   "L2 Losses"      : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)};

    # Epochs!!!
    print("\nRunning %d epochs..." % Settings["Num Epochs"]);
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
                                Weights     = Settings["Weights"],
                                Optimizer   = Optimizer,
                                Device      = Settings["Device"]);

        # Update the loss history buffers.
        Train_Losses["Data Losses"][t]  = Train_Dict["Data Loss"];
        Train_Losses["Coll Losses"][t]  = Train_Dict["Coll Loss"];
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
                                Weights     = Settings["Weights"],
                                Device      = Settings["Device"]);

        Test_Losses["Data Losses"][t]   = Test_Dict["Data Loss"];
        Test_Losses["Coll Losses"][t]   = Test_Dict["Coll Loss"];
        Test_Losses["Total Losses"][t]  = Test_Dict["Total Loss"];

        # Now record the parameter losses.
        Parameter_Losses["Lp Losses"][t]    = Test_Dict["Lp Loss"];
        Parameter_Losses["L2 Losses"][t]    = Test_Dict["L2 Loss"];


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
            print("            | Train:\t Data = %.7f\t Coll = %.7f\t Total = %.7f" % (Train_Dict["Data Loss"], Train_Dict["Coll Loss"], Train_Dict["Total Loss"]));
            print("            | Test: \t Data = %.7f\t Coll = %.7f\t Total = %.7f" % (Test_Dict["Data Loss"], Test_Dict["Coll Loss"], Test_Dict["Total Loss"]));
            print("Epoch #%-4d |       \t Lp   = %.7f\t L2   = %.7f" % (t, Test_Dict["Lp Loss"], Test_Dict["L2 Loss"]))
        else:
            print("Epoch #%-4d | \t\t Targeted %3d \t\t Cutoff = %g"   % (t, Targeted_Coll_Pts.shape[0], Cutoff));

    # Report runtime!
    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs, an average of %7.2fs per epoch)" % (Epoch_Runtime,  (Epoch_Runtime / Settings["Num Epochs"])));


    ############################################################################
    # Threshold Xi.

    # Recall how we enforce Xi: We trick torch into minimizing
    #       [Xi]_1^p + ... + [Xi]_n^p,
    # which is highly concave (for p < 1), by instead minimizing
    #       w_1[Xi]_1^2 + ... + w_n[Xi]_n^2,
    # where each w_i is updated each step to be w_i = [Xi]_i^{2 - p}. The above
    # is convex (if we treat w_1, ... , w_n as constants). There is, however, a
    # problem. If [Xi]_i is smaller than about 3e-4, then [Xi]_i^2 is roughly
    # machine Epsilon, meaning we run into problems. To avoid this, we instead
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
    print();
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

    print("\nSaving...", end = '');

    # First, come up with a save name that does not conflict with an existing
    # save name. To do this, we first attempt to make a save file name that
    # consists of the data set name plus the activation function and optimizer
    # we used. If a save with that name already exists, we append a "1" to the
    # end of the file name. If that also corresponds to an existing save, then
    # we replace the "1" with a "2" and so on until we get save name that does
    # not already exist.
    Base_File_Name : str = Settings["DataSet Name"] + "_" + Settings["Hidden Activation Function"] + "_" + Settings["Optimizer"]

    Counter         : int = 0;
    Save_File_Name  : str = Base_File_Name;
    while(os.path.isfile("../Saves/" + Save_File_Name)):
        # Increment the counter, try appending that onto Base_File_Name.
        Counter         += 1;
        Save_File_Name   = Base_File_Name + ("_%u" % Counter);

    # Next, get the encoding vectors for each element of Derivatives.
    Derivative_Encodings : List[numpy.ndarray] = [];
    for i in range(len(Settings["Derivatives"])):
        Derivative_Encodings.append(Settings["Derivatives"][i].Encoding)

    # Finally, get the state of each library term.
    RHS_Term_States : List[Dict] = [];
    for i in range(len(Settings["RHS Terms"])):
        RHS_Term_States.append(Settings["RHS Terms"][i].Get_State());

    LHS_Term_State : Dict = Settings["LHS Term"].Get_State();    

    # We can now save!
    torch.save({"U State"               : U.Get_State(),
                "Xi"                    : Xi,
                "Optimizer"             : Optimizer.state_dict(),
                "Derivative Encodings"  : Derivative_Encodings,
                "LHS Term State"        : LHS_Term_State,
                "RHS Term States"       : RHS_Term_States},
                "../Saves/" + Save_File_Name);

    print("Done! Saved as \"%s\"" % Save_File_Name);


    ############################################################################
    # Plot. 

    Plot_Losses(Save_File_Name      = Save_File_Name,
                Train_Losses        = [Train_Losses],
                Test_Losses         = [Test_Losses],
                Parameter_Losses    = [Parameter_Losses],
                Labels              = [Save_File_Name]);
                



if(__name__ == "__main__"):
    main();
