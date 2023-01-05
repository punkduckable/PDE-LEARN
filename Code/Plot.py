import  numpy;
from    matplotlib  import  pyplot  as plt;
import  seaborn;
import  pandas;
import  os;
from    typing      import Dict, List;



def Plot_Losses(Save_File_Name      : str,
                Train_Losses        : List[Dict[str, numpy.ndarray]],
                Test_Losses         : List[Dict[str, numpy.ndarray]],
                L2_Losses           : List[numpy.ndarray],
                Lp_Losses           : numpy.ndarray,
                Labels              : List[str]) -> None:
    """
    This function plots loss histories. 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Save_File_Name: This function makes a new directory to house the figures. "Save_File_Name" is
    the name of that directory.

    Train/Test_Losses: A list of dictionaries. The ith dictionary should house a dictionary with 
    the test/train loss histories of the ith solution network.

    L2_Losses: A list of numpy arrays whose ith entry holds the L2 loss history for the ith 
    solution network.

    Lp_Losses: A numpy ndarray whose ith entry holds the Lp loss from the ith epoch.

    Labels: A List of strings. We use this to label the plots.
    """

    assert(len(Train_Losses) == len(Test_Losses));
    assert(len(Train_Losses) == len(L2_Losses));
    assert(len(Train_Losses) == len(Labels));

    ###############################################################################################
    # Setup.

    # Make a folder to house the figures.
    Plot_Directory_Name : str = "Loss_History_" + Save_File_Name;
    Plot_Directory_Path : str = "../Figures/"   + Plot_Directory_Name;
    os.mkdir(Plot_Directory_Path);

    # Extract each loss type for each experiment.
    Num_Experiments : int = len(Labels);

    Train_Total_Losses  : List[numpy.ndarray] = [];
    Train_Data_Losses   : List[numpy.ndarray] = [];
    Train_Coll_Losses   : List[numpy.ndarray] = [];

    Test_Total_Losses   : List[numpy.ndarray] = [];
    Test_Data_Losses    : List[numpy.ndarray] = [];
    Test_Coll_Losses    : List[numpy.ndarray] = [];

    for i in range(Num_Experiments):
        Train_Total_Losses  .append(Train_Losses[i]["Total Losses"]);
        Train_Data_Losses   .append(Train_Losses[i]["Data Losses"]);
        Train_Coll_Losses   .append(Train_Losses[i]["Coll Losses"]);

        Test_Total_Losses   .append(Test_Losses[i]["Total Losses"]);        
        Test_Data_Losses    .append(Test_Losses[i]["Data Losses"]);
        Test_Coll_Losses    .append(Test_Losses[i]["Coll Losses"]);

    # Generate data frames.
    Total_DF : pandas.DataFrame = Make_Test_Train_DataFrame(
                                    Train_Losses        = Train_Total_Losses,
                                    Test_Losses         = Test_Total_Losses,
                                    Labels              = Labels);

    Data_DF : pandas.DataFrame = Make_Test_Train_DataFrame(
                                    Train_Losses        = Train_Data_Losses,
                                    Test_Losses         = Test_Data_Losses,
                                    Labels              = Labels);

    Coll_DF : pandas.DataFrame = Make_Test_Train_DataFrame(
                                    Train_Losses        = Train_Coll_Losses,
                                    Test_Losses         = Test_Coll_Losses,
                                    Labels              = Labels);
   
    Lp_DF : pandas.DataFrame = Make_Parameter_DataFrame(
                                    Parameter_Losses    = [Lp_Losses],
                                    Labels              = [""]);

    L2_DF : pandas.DataFrame = Make_Parameter_DataFrame(
                                    Parameter_Losses    = L2_Losses,
                                    Labels              = Labels);

    # Pick seaborn settings.
    seaborn.set_context(context     = "paper");
    seaborn.set_style(  style       = "darkgrid");
    palette : str = "winter";


    ###############################################################################################
    # Total loss.

    plt.figure(0);
    plt.clf();
    seaborn.lineplot(data = Total_DF, x = "Epoch Number", y = "Losses", hue = "Experiment", style = "Train/Test", palette = palette);
    plt.title("Total Loss");
    plt.xlabel("epoch number");
    plt.ylabel("Loss");
    plt.yscale('log');
    plt.savefig(Plot_Directory_Path + "/Total_Loss.png", dpi = 500);

    # Make a figure for the history of the Data loss.
    plt.figure(1);
    plt.clf();
    seaborn.lineplot(data = Data_DF, x = "Epoch Number", y = "Losses", hue = "Experiment", style = "Train/Test", palette = palette);
    plt.title("Data Loss");
    plt.xlabel("epoch number");
    plt.ylabel("Loss");
    plt.yscale('log');
    plt.legend();
    plt.savefig(Plot_Directory_Path + "/Data_Loss.png", dpi = 500);

    # Make a figure for the history of the Collocation loss.
    plt.figure(2);
    plt.clf();
    seaborn.lineplot(data = Coll_DF, x = "Epoch Number", y = "Losses", hue = "Experiment", style = "Train/Test", palette = palette);
    plt.title("Collocation Loss");
    plt.xlabel("epoch number");
    plt.ylabel("Loss");
    plt.yscale('log');
    plt.legend();
    plt.savefig(Plot_Directory_Path + "/Coll_Loss.png", dpi = 500);

    # Make a figure for the history of the Lp loss.
    plt.figure(3);
    plt.clf();
    seaborn.lineplot(data = Lp_DF, x = "Epoch Number", y = "Losses", hue = "Experiment", palette = palette);
    plt.title("Lp Loss")
    plt.xlabel("epoch number");
    plt.ylabel("Loss");
    plt.savefig(Plot_Directory_Path + "/Lp_Loss.png", dpi = 500);

    # Make a figure for the history of the L2 loss.
    plt.figure(4);
    plt.clf();
    seaborn.lineplot(data = L2_DF, x = "Epoch Number", y = "Losses", hue = "Experiment", palette = palette);
    plt.title("L2 Loss");
    plt.xlabel("epoch number");
    plt.ylabel("Loss");
    plt.savefig(Plot_Directory_Path + "/L2_Loss.png", dpi = 500);

    # All done... reveal the plots!
    plt.show();



def Make_Parameter_DataFrame(   Parameter_Losses    : List[numpy.ndarray],
                                Labels              : List[str]) -> pandas.DataFrame:
    """
    This function takes lists containing Parameter losses (Xi or L2) for multiple experiments and
    packages them together into a DataFrame object that is suitable for plotting.

    -----------------------------------------------------------------------------------------------
    Arguments:

    Parameter_Losses: This is a list of numpy arrays. The ith element should be a 1D numpy ndarray
    that holds the loss history for some experiment.

    Labels: This is a list with one label per experiment. We use these to label the entries in
    the returned DataFrame.

    -----------------------------------------------------------------------------------------------
    Returns:

    A DataFrame object. This object has three columns: "Losses", "Epoch Number", and
    "Label". The "Losses" holds all the elements from Train_Losses and Test_Losses as a single,
    1D column matrix. Obviously, by doing this we lose some information (like which experiment or
    epoch number a given entry of the "Losses" column belongs to). The other three columns give
    us this extra information. The ith entry of "Experiment" specifies which experiment (column of
    Train/Test_Losses) the ith entry of "Losses" belongs to. Finally, the ith entry of "Epoch Number"
    specifies the epoch number (from a particular experiment) when we recorded the value in the ith
    entry of "Losses".
    """

    assert(len(Parameter_Losses) == len(Labels));

    # First, determine tht total number of data points in Parameter_Losses.
    Num_Experiments         : int = len(Parameter_Losses);
    Num_Data_Points         : int = 0;

    for i in range(Num_Experiments):
        Num_Data_Points += Parameter_Losses[i].size;

    # We will build up four columns: one for the losses, one that specifies if a given row is a
    # test or train measurement, one that specifies which experiment a given row belongs to,
    # and one that specifies the epoch number when each loss was originally recorded.
    Losses              : numpy.ndarray = numpy.empty(shape = Num_Data_Points, dtype = numpy.float32);
    Row_Epoch_Number    : numpy.ndarray = numpy.empty(shape = Num_Data_Points, dtype = numpy.int32);
    Row_Labels          : List[str]     = [];

    Index = 0;
    for i in range(Num_Experiments):
        Num_Points_ith_Experiment : int = Parameter_Losses[i].size;

        # Add the parameter losses from this experiment.
        Losses[Index:(Index + Num_Points_ith_Experiment)] = Parameter_Losses[i];
        Row_Epoch_Number[Index:(Index + Num_Points_ith_Experiment)] = numpy.arange(start = 0, stop = Num_Points_ith_Experiment);

        Row_Labels          = Row_Labels        + Num_Points_ith_Experiment*[Labels[i]];

        Index += Num_Points_ith_Experiment;

    # We can now make a data frame from these three columns.
    return pandas.DataFrame({   "Losses"        : Losses,
                                "Epoch Number"  : Row_Epoch_Number,
                                "Experiment"    : Row_Labels});





def Make_Test_Train_DataFrame(  Train_Losses        : List[numpy.ndarray],
                                Test_Losses         : List[numpy.ndarray],
                                Labels              : List[str]) -> pandas.DataFrame:
    """
    This function takes lists containing Test/Train losses for multiple experiments and packages
    them together into a DataFrame object that is suitable for plotting.

    -----------------------------------------------------------------------------------------------
    Arguments:

    Train_Losses: This is a list of numpy arrays. The ith element should be a 1D numpy ndarray
    that holds the loss history for the training set in some experiment. Critically, the length
    of the ith element should match the length of the ith element of Test_Losses.

    Test_Losses: This is a list of numpy arrays. The ith element should be a 1D numpy ndarray
    that holds the loss history for the testing set in some experiment. Critically, the length
    of the ith element should match the length of the ith element of Train_Losses.

    Labels: This is a list with one label per experiment. We use these to label the entries in
    the returned DataFrame.

    -----------------------------------------------------------------------------------------------
    Returns:

    A DataFrame object. This object has four columns: "Losses", "Epoch Number", "Train/Test", and
    "Label". The "Losses" holds all the elements from Train_Losses and Test_Losses as a single,
    1D column matrix. Obviously, by doing this we lose some information (like which experiment or
    epoch number a given entry of the "Losses" column belongs to). The other three columns give
    us this extra information. The ith entry of "Test/Train" specifies if the ith entry of
    "Losses" represents the loss of a testing or training set. The ith entry of "Experiment"
    specifies which experiment (column of Train/Test_Losses) the ith entry of "Losses" belongs to.
    Finally, the ith entry of "Epoch Number" specifies the epoch number (from a particular
    experiment) when we recorded the value in the ith entry of "Losses".
    """

    assert(len(Train_Losses) == len(Test_Losses));
    assert(len(Train_Losses) == len(Labels));
    for i in range(len(Train_Losses)):
        assert(Train_Losses[i].shape == Test_Losses[i].shape);

    # First, determine the total number of data points in Train/Test Losses.
    Num_Experiments         : int = len(Train_Losses);
    Num_Data_Points         : int = 0;

    for i in range(Num_Experiments):
        Num_Data_Points += Train_Losses[i].size;

    # We will build up four columns: one for the losses, one that specifies if a given row is a
    # test or train measurement, one that specifies which experiment a given row belongs to,
    # and one that specifies the epoch number when each loss was originally recorded.
    Losses              : numpy.ndarray = numpy.empty(shape = 2*Num_Data_Points, dtype = numpy.float32);
    Row_Epoch_Number    : numpy.ndarray = numpy.empty(shape = 2*Num_Data_Points, dtype = numpy.int32);
    Row_Train_Test      : List[str]     = [];
    Row_Labels          : List[str]     = [];

    Index = 0;
    for i in range(Num_Experiments):
        Num_Points_ith_Experiment : int = Train_Losses[i].size;

        # Add the training losses from this experiment.
        Losses[Index:(Index + Num_Points_ith_Experiment)] = Train_Losses[i];
        Row_Epoch_Number[Index:(Index + Num_Points_ith_Experiment)] = numpy.arange(start = 0, stop = Num_Points_ith_Experiment);

        Row_Train_Test      = Row_Train_Test    + Num_Points_ith_Experiment*["Train"];
        Row_Labels          = Row_Labels        + Num_Points_ith_Experiment*[Labels[i]];

        Index += Num_Points_ith_Experiment;

        # Add the testing losses from this series
        Losses[Index:(Index + Num_Points_ith_Experiment)] = Test_Losses[i];
        Row_Epoch_Number[Index:(Index + Num_Points_ith_Experiment)] = numpy.arange(start = 0, stop = Num_Points_ith_Experiment);

        Row_Train_Test  = Row_Train_Test    + Num_Points_ith_Experiment*["Test"];
        Row_Labels      = Row_Labels        + Num_Points_ith_Experiment*[Labels[i]];

        Index += Num_Points_ith_Experiment;

    # We can now make a data frame from these three columns.
    return pandas.DataFrame({   "Losses"        : Losses,
                                "Epoch Number"  : Row_Epoch_Number,
                                "Train/Test"    : Row_Train_Test,
                                "Experiment"    : Row_Labels});



