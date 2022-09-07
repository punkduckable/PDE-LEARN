# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       : str = os.path.abspath(os.path.curdir);
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
        print("%-25s = %s" % (Setting, str(Value)), end = '');

    # Start a setup timer.
    Setup_Timer : float = time.perf_counter();
    print("\nSetting up...\n");


    ############################################################################
    # Set up Data
    # This sets up the testing/training inputs/targets, and the input bounds.
    # This will also tell us the number of spatial dimensions (there's a row of
    # Input_Bounds for each coordinate component. Since one coordinate is for
    # time, one minus the number of rows gives the number of spatial dimensions).

    Num_DataSets    : int                       = len(Settings["DataSet Names"]);
    Data_Dict       : Dict[str, numpy.ndarray]  = { "Train Inputs"          : [],
                                                    "Train Targets"         : [],
                                                    "Test Inputs"           : [],
                                                    "Test Targets"          : [],
                                                    "Input Bounds"          : [],
                                                    "Number of Dimensions"  : []};
    for i in range(Num_DataSets):
        ith_Data_Dict : Dict = Data_Loader( DataSet_Name    = Settings["DataSet Names"][i],
                                            Device          = Settings["Device"]);
        
        Data_Dict["Train Inputs"            ].append(ith_Data_Dict["Train Inputs"]);
        Data_Dict["Train Targets"           ].append(ith_Data_Dict["Train Targets"]);
        Data_Dict["Test Inputs"             ].append(ith_Data_Dict["Test Inputs"]);
        Data_Dict["Test Targets"            ].append(ith_Data_Dict["Test Targets"]);
        Data_Dict["Input Bounds"            ].append(ith_Data_Dict["Input Bounds"]);
        Data_Dict["Number of Dimensions"    ].append(ith_Data_Dict["Input Bounds"].shape[0]);

    # Now determine the number of dimensions of the data. This should be the 
    # same for each data set.
    Num_Dimensions : int   = Data_Dict["Number of Dimensions"][0];
    for i in range(1, Num_DataSets):
        assert(Data_Dict["Number of Dimensions"][i] == Num_Dimensions);


    ############################################################################
    # Set up U for each data set, as well as the common Xi, Library

    # First, if we are loading anything, load in the save.
    if( Settings["Load U"]              == True or 
        Settings["Load Xi, Library"]    == True or
        Settings["Load Optimizer"]      == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings["Load File Name"];
        Saved_State = torch.load(Load_File_Path, map_location = Settings["Device"]);


    # First, either build U or load its state from save. 
    if(Settings["Load U"] == True):
        # Fetch each solution's state. There should be one state for each data set.
        U_States : List[Dict] = Saved_State["U States"];
        assert(len(U_States) == Num_DataSets);

        U_List : List[Network] = [];
        for i in range(Num_DataSets):
            # First, fetch Widths and Activation functions from U[i]'s State.
            Ui_State            : Dict      = U_States[i];
            Widths              : List[int] = Ui_State["Widths"];
            Hidden_Activation   : str       = Ui_State["Activation Types"][0];
            Output_Activation   : str       = Ui_State["Activation Types"][-1];

            Settings["Hidden Activation Function"]  = Hidden_Activation;

            # Set up U[i].
            U_List.append(Network(  Widths              = Widths, 
                                    Hidden_Activation   = Hidden_Activation, 
                                    Output_Activation   = Output_Activation,
                                    Device              = Settings["Device"]));
            U_List[i].Set_State(Ui_State);

            # Report!
            print("Loaded U[%u] from state." % i);
            print("    Hidden Activation:   %s" % Hidden_Activation);
            print("    Widths:              %s" % str(Widths), end = '');

    else:
        # First, set up Widths. This is an array whose ith entry specifies the width of 
        # the ith layer of the network (including the input and output layers).
        Widths = [Num_Dimensions] + Settings["Hidden Layer Widths"] + [1];
        
        # Now initialize each U[i].
        U_List : List[Network] = [];
        for i in range(Num_DataSets):
            U_List.append(Network(   Widths              = Widths,
                                Hidden_Activation   = Settings["Hidden Activation Function"],
                                Device              = Settings["Device"]));
        
        print("Set up the solution networks using settings in Settings.txt.")


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

    Params = [];
    for i in range(Num_DataSets):
        Params = Params + list(U_List[i].parameters());
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
    Targeted_Coll_Pts_List : List[torch.Tensor]= [];
    for i in range(Num_DataSets):
        Targeted_Coll_Pts_List.append(torch.empty((0, Num_Dimensions), dtype = torch.float32));

    # Set up buffers to hold losses, also set up a timer.
    Epoch_Timer         : float                             = time.perf_counter();
    Train_Losses        : List[Dict[str, numpy.ndarray]]    = [];
    Test_Losses         : List[Dict[str, numpy.ndarray]]    = [];
    L2_Losses           : List[numpy.ndarray]               = [];
    Lp_Losses           : numpy.ndarray                     = numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32);

    for i in range(Num_DataSets):
        Train_Losses.append({   "Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)});

        Test_Losses.append({    "Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Coll Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)});

        L2_Losses.append(numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32));

    # Epochs!!!
    print("\nRunning %d epochs..." % Settings["Num Epochs"]);
    for t in range(0, Settings["Num Epochs"]):
        ########################################################################
        # Train

        # First, we need to set up the collocation points for each data set for
        # this epoch. This set is a combination of randomly generated points 
        # and the targeted points from the last epoch.

        Train_Coll_Points_List : List[numpy.ndarray] = [];
        for i in range(Num_DataSets):
            ith_Random_Coll_Points  : torch.Tensor = Generate_Points(
                                                    Bounds      = Data_Dict["Input Bounds"][i],
                                                    Num_Points  = Settings["Num Train Coll Points"],
                                                    Device      = Settings["Device"]);

            Train_Coll_Points_List.append(torch.vstack((ith_Random_Coll_Points, Targeted_Coll_Pts_List[i])));

        # Now run a Training Epoch.
        Train_Dict = Training(  U_List              = U_List,
                                Xi                  = Xi,
                                Coll_Points_List    = Train_Coll_Points_List,
                                Inputs_List         = Data_Dict["Train Inputs"],
                                Targets_List        = Data_Dict["Train Targets"],
                                Derivatives         = Settings["Derivatives"],
                                LHS_Term            = Settings["LHS Term"],
                                RHS_Terms           = Settings["RHS Terms"],
                                p                   = Settings["p"],
                                Weights             = Settings["Weights"],
                                Optimizer           = Optimizer,
                                Device              = Settings["Device"]);

        # Append the train loss history.
        for i in range(Num_DataSets):
            Train_Losses[i]["Data Losses"][t]  = Train_Dict["Data Losses"][i];
            Train_Losses[i]["Coll Losses"][t]  = Train_Dict["Coll Losses"][i];
            Train_Losses[i]["Total Losses"][t] = Train_Dict["Total Losses"][i];


        ########################################################################
        # Test

        # First, generate random collocation points then evaluate the
        # network on them.
        Test_Coll_Points_List : List[torch.Tensor]= [];
        for i in range(Num_DataSets):
            Test_Coll_Points_List.append(Generate_Points(
                                Bounds      = Data_Dict["Input Bounds"][i],
                                Num_Points  = Settings["Num Test Coll Points"],
                                Device      = Settings["Device"]));

        # Evaluate losses on the testing points.
        Test_Dict = Testing(    U_List              = U_List,
                                Xi                  = Xi,
                                Coll_Points_List    = Test_Coll_Points_List,
                                Inputs_List         = Data_Dict["Test Inputs"],
                                Targets_List        = Data_Dict["Test Targets"],
                                Derivatives         = Settings["Derivatives"],
                                LHS_Term            = Settings["LHS Term"],
                                RHS_Terms           = Settings["RHS Terms"],
                                p                   = Settings["p"],
                                Weights             = Settings["Weights"],
                                Device              = Settings["Device"]);

        # Append the test loss history.
        for i in range(Num_DataSets):
            Test_Losses[i]["Data Losses"][t]   = Test_Dict["Data Losses"][i];
            Test_Losses[i]["Coll Losses"][t]   = Test_Dict["Coll Losses"][i];
            Test_Losses[i]["Total Losses"][t]  = Test_Dict["Total Losses"][i];

        # Now record the Lp, L2 losses.
        Lp_Losses[t] = Test_Dict["Lp Loss"];

        for i in range(Num_DataSets):
            L2_Losses[i][t] = Test_Dict["L2 Losses"][i];


        ########################################################################
        # Update targeted residual points.

        for i in range(Num_DataSets):
            # Find the Absolute value of the residuals for the ith data set. 
            # Isolate those corresponding to the "random" collocation points.
            Abs_Residual        : torch.Tensor = torch.abs(Train_Dict["Residuals"][i]);
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
            Targeted_Coll_Pts_List[i] = Train_Coll_Points_List[i][Big_Residual_Indices, :].detach();


        ########################################################################
        # Report!

        if(t % 10 == 0 or t == Settings["Num Epochs"] - 1):
            for i in range(Num_DataSets):
                print("            |");
                print("            | Train:\t Data[%u] = %.7f\t Coll[%u] = %.7f\t Total[%u] = %.7f" % (i, Train_Dict["Data Losses"][i], i, Train_Dict["Coll Losses"][i], i, Train_Dict["Total Losses"][i]));
                print("            | Test: \t Data[%u] = %.7f\t Coll[%u] = %.7f\t Total[%u] = %.7f" % (i, Test_Dict["Data Losses"][i], i, Test_Dict["Coll Losses"][i], i, Test_Dict["Total Losses"][i]));
                if(i == 0):
                    print("Epoch #%-4d |       \t Lp      = %.7f\t L2[%u]   = %.7f" % (t + 1, Test_Dict["Lp Loss"], i, Test_Dict["L2 Losses"][i]));
                else: 
                    print("            |       \t Lp      = %.7f\t L2[%u]   = %.7f" % (Test_Dict["Lp Loss"], i, Test_Dict["L2 Losses"][i]));
                print("            |");
        else:
            print("Epoch #%-4d | \t" % (t + 1), end = '');
            for i in range(Num_DataSets):
                print("\t Targeted[%u] = %3d \t Cutoff[%u] = %.7f "   % (i, Targeted_Coll_Pts_List[i].shape[0], i, Cutoff), end = '');
            print();

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
    Base_File_Name  : str = "";
    for i in range(Num_DataSets):
        Base_File_Name += Settings["DataSet Names"][i] + "_";
    Base_File_Name  += Settings["Hidden Activation Function"] + "_" + Settings["Optimizer"];

    Counter         : int = 0;
    Save_File_Name  : str = Base_File_Name;
    while(os.path.isfile("../Saves/" + Save_File_Name)):
        # Increment the counter, try appending that onto Base_File_Name.
        Counter         += 1;
        Save_File_Name   = Base_File_Name + ("_%u" % Counter);

    # Next, get each U's state
    U_States : List[Dict] = [];
    for i in range(Num_DataSets):
        U_States.append(U_List[i].Get_State());

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
    torch.save({"U States"              : U_States,
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
                Train_Losses        = Train_Losses,
                Test_Losses         = Test_Losses,
                L2_Losses           = L2_Losses,
                Lp_Losses           = Lp_Losses,
                Labels              = Settings["DataSet Names"]);
                



if(__name__ == "__main__"):
    main();
