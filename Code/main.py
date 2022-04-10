import numpy;
import torch;
import time;

from Settings_Reader import Settings_Reader, Settings_Container;
from Mappings import    Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, \
                        Num_Sub_Index_Values_1D, Num_Sub_Index_Values_2D, \
                        Max_Col_Num, \
                        Col_Number_to_Multi_Index_Class;
from Loss           import Data_Loss, Lp_Loss, Coll_Loss;
from Network        import Rational, Neural_Network;
from Data_Loader    import Data_Loader;
from Test_Train     import Testing, Training;
from Points         import Generate_Points;
from Xi             import Print_PDE;



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


    ############################################################################
    # Determine the number of index values, library terms.

    # Determine how many index values we can have. This value will be important
    # going forward.
    Num_Sub_Index_Values : int = 0;
    if(Settings.Num_Spatial_Dimensions == 1):
        Num_Sub_Index_Values = Num_Sub_Index_Values_1D(Settings.Highest_Order_Spatial_Derivatives);
    if(Settings.Num_Spatial_Dimensions == 2):
        Num_Sub_Index_Values = Num_Sub_Index_Values_2D(Settings.Highest_Order_Spatial_Derivatives);


    # Now, determine how many library terms we have. This will determine the
    # size of Xi.
    Num_Library_Terms : int = Max_Col_Num(Max_Sub_Indices      = Settings.Maximum_Term_Degree,
                                          Num_Sub_Index_Values = Num_Sub_Index_Values);


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
    Xi = torch.zeros(   Num_Library_Terms + 1,
                        dtype           = torch.float32,
                        device          = Settings.Device,
                        requires_grad   = True);


    ############################################################################
    # Set up the Col_Number_to_Multi_Index map.

    Col_Number_to_Multi_Index = Col_Number_to_Multi_Index_Class(
                                    Max_Sub_Indices      = Settings.Maximum_Term_Degree,
                                    Num_Sub_Index_Values = Num_Sub_Index_Values);


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



    ############################################################################
    # Set up Index_to_Derivatives.

    if(Settings.Num_Spatial_Dimensions == 1):
        Index_to_Derivatives = Index_to_x_Derivatives;
    if(Settings.Num_Spatial_Dimensions == 2):
        Index_to_Derivatives = Index_to_xy_Derivatives_Class(
                                        Highest_Order_Derivatives   = Settings.Highest_Order_Spatial_Derivatives);

    # Setup is now complete. Report time.
    print("Done! Took %7.2fs" % (time.perf_counter() - Setup_Timer));


    ############################################################################
    # Run the Epochs!

    Epoch_Timer : float = time.perf_counter();
    print("Running %d epochs..." % Settings.Num_Epochs);

    for t in range(Settings.Num_Epochs):
        # First, generate new training collocation points.
        Train_Coll_Points = Generate_Points(
                        Bounds      = Data_Container.Input_Bounds,
                        Num_Points  = Settings.Num_Train_Coll_Points,
                        Device      = Settings.Device);

        # Now run a Training Epoch.
        Training(   U                                   = U,
                    Xi                                  = Xi,
                    Coll_Points                         = Train_Coll_Points,
                    Inputs                              = Data_Container.Train_Inputs,
                    Targets                             = Data_Container.Train_Targets,
                    Time_Derivative_Order               = Settings.Time_Derivative_Order,
                    Highest_Order_Spatial_Derivatives   = Settings.Highest_Order_Spatial_Derivatives,
                    Index_to_Derivatives                = Index_to_Derivatives,
                    Col_Number_to_Multi_Index           = Col_Number_to_Multi_Index,
                    p                                   = Settings.p,
                    Lambda                              = Settings.Lambda,
                    Optimizer                           = Optimizer,
                    Device                              = Settings.Device);

        # Test the code (and print the loss) every 10 Epochs. For all other
        # epochs, print the Epoch to indicate the program is making progress.
        if(t % 10 == 0 or t == Settings.Num_Epochs - 1):
            # Generate new testing Collocation Coordinates
            Test_Coll_Points = Generate_Points(
                            Bounds      = Data_Container.Input_Bounds,
                            Num_Points  = Settings.Num_Test_Coll_Points,
                            Device      = Settings.Device);

            # Evaluate losses on training points.
            (Train_Data_Loss, Train_Coll_Loss, Train_Lp_Loss) = Testing(
                U                                   = U,
                Xi                                  = Xi,
                Coll_Points                         = Train_Coll_Points,
                Inputs                              = Data_Container.Train_Inputs,
                Targets                             = Data_Container.Train_Targets,
                Time_Derivative_Order               = Settings.Time_Derivative_Order,
                Highest_Order_Spatial_Derivatives   = Settings.Highest_Order_Spatial_Derivatives,
                Index_to_Derivatives                = Index_to_Derivatives,
                Col_Number_to_Multi_Index           = Col_Number_to_Multi_Index,
                p                                   = Settings.p,
                Lambda                              = Settings.Lambda,
                Device                              = Settings.Device);

            # Evaluate losses on the testing points.
            (Test_Data_Loss, Test_Coll_Loss, Test_Lp_Loss) = Testing(
                U                                   = U,
                Xi                                  = Xi,
                Coll_Points                         = Test_Coll_Points,
                Inputs                              = Data_Container.Test_Inputs,
                Targets                             = Data_Container.Test_Targets,
                Time_Derivative_Order               = Settings.Time_Derivative_Order,
                Highest_Order_Spatial_Derivatives   = Settings.Highest_Order_Spatial_Derivatives,
                Index_to_Derivatives                = Index_to_Derivatives,
                Col_Number_to_Multi_Index           = Col_Number_to_Multi_Index,
                p                                   = Settings.p,
                Lambda                              = Settings.Lambda,
                Device                              = Settings.Device);

            # Print losses!
            print("Epoch #%-4d | Test: \t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (t, Test_Data_Loss, Test_Coll_Loss, Test_Lp_Loss, Test_Data_Loss + Test_Coll_Loss + Test_Lp_Loss));
            print("            | Train:\t Data = %.7f\t Coll = %.7f\t Lp = %.7f \t Total = %.7f"
                % (Train_Data_Loss, Train_Coll_Loss, Train_Lp_Loss, Train_Data_Loss + Train_Coll_Loss + Train_Lp_Loss));
        else:
            print(("Epoch #%-4d | "   % t));

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

    Print_PDE(  Xi                        = Pruned_Xi,
                Time_Derivative_Order     = Settings.Time_Derivative_Order,
                Num_Spatial_Dimensions    = Settings.Num_Spatial_Dimensions,
                Index_to_Derivatives      = Index_to_Derivatives,
                Col_Number_to_Multi_Index = Col_Number_to_Multi_Index);


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
