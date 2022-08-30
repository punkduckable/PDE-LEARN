import  torch;
from    typing  import Dict, List;

from File_Reader    import Read_Error, Read_Line_After, Read_Bool_Setting, Read_Setting, Read_List_Setting, Read_Dict_Setting;
from Library_Reader import Read_Library;



def Settings_Reader() -> Dict:
    """ This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A dictionary housing the settings we read from Settings.txt. The main
    function uses these to set up the program. """

    File = open("../Settings.txt", "r");
    Settings = {};


    ############################################################################
    # Save, Load Settings

    # Load Sol, Xi, or Optimizer from File?
    Settings["Load U"]          = Read_Bool_Setting(File, "Load U From Save [bool]:");
    Settings["Load Xi"]         = Read_Bool_Setting(File, "Load Xi From Save [bool]:");
    Settings["Load Optimizer"]  = Read_Bool_Setting(File, "Load Optimizer From Save [bool]:");

    # If so, get the load file name.
    if( Settings["Load U"]          == True or
        Settings["Load Xi"]         == True or
        Settings["Load Optimizer"]  == True):

        Settings["Load File Name"] = Read_Line_After(File, "Load File Name [str]:").strip();


    ############################################################################
    # Library Settings.

    # Where is the file that lists the library functions / derivatives?
    Library_File_Name : str             = Read_Setting(File, "Library File [str]:");
    Library_Path      : str             = "../" + Library_File_Name + ".txt";
    Derivatives, LHS_Term, RHS_Terms    = Read_Library(Library_Path);

    Settings["Derivatives"] = Derivatives;
    Settings["LHS Term"]    = LHS_Term;
    Settings["RHS Terms"]   = RHS_Terms;



    ############################################################################
    # Network settings.

    # Width of the hidden layers
    Buffer : List[str] = Read_List_Setting(File, "Hidden Layer Widths [List of int]:");
    for i in range(len(Buffer)):
        Buffer[i] = int(Buffer[i]);
    
    Settings["Hidden Layer Widths"] = Buffer;

    # Which activation function should we use?
    Buffer = Read_Setting(File, "Activation Function [Tanh, Rational, Sin]:");
    if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
        Settings["Activation Function"] = "Rational";
    elif(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings["Activation Function"] = "Tanh";
    elif(Buffer[0] == 'S' or Buffer[0] == 's'):
        Settings["Activation Function"] = "Sin";
    else:
        raise Read_Error("\"Activation Function [Tanh, Rational, Sin]:\" should be" + \
                         "\"Tanh\", \"Rational\", or \"Sin\" Got " + Buffer);

    Buffer = Read_Setting(File, "Train on CPU or GPU [GPU, CPU]:");
    if(Buffer[0] == 'G' or Buffer[0] == 'g'):
        if(torch.cuda.is_available() == True):
            Settings["Device"] = torch.device('cuda');
        else:
            Settings["Device"] = torch.device('cpu');
            print("You requested a GPU, but cuda is not available on this machine. Switching to CPU");
    elif(Buffer[0] == 'C' or Buffer[0] == 'c'):
        Settings["Device"] = torch.device('cpu');
    else:
        raise Read_Error("\"Train on CPU or GPU\" should be \"CPU\" or \"GPU\". Got " + Buffer);



    ############################################################################
    # Loss settings.

    # Read p values (used in the Lp loss function).
    Settings["p"]      = float(Read_Setting(File, "p [float]:"));

    # Read weights (used to scale the components of the loss function)
    Buffer : Dict[str, str] = Read_Dict_Setting(File, "Weights [Dict of float]:");

    for (key, value) in Buffer.items():
        Buffer[key] = float(value);

    Settings["Weights"] = Buffer;

    # Read number of testing/training data/collocation points
    Settings["Num Train Coll Points"]   = int(Read_Setting(File, "Number of Training Collocation Points [int]:"));
    Settings["Num Test Coll Points"]    = int(Read_Setting(File, "Number of Testing Collocation Points [int]:"));



    ############################################################################
    # Optimizer settings.

    # Read the optimizer type.
    Buffer = Read_Setting(File, "Optimizer [Adam, LBFGS]:");
    if  (Buffer[0] == 'A' or Buffer[0] == 'a'):
        Settings["Optimizer"] = "Adam";
    elif(Buffer[0] == 'L' or Buffer[0] == 'l'):
        Settings["Optimizer"] = "LBFGS";
    else:
        raise Read_Error("\"Optimizer [Adam, LBFGS]:\" should be \"Adam\" or \"LBFGS\". Got " + Buffer);

    # Read the learning rate, number of epochs.
    Settings["Learning Rate"] = float(Read_Setting(File, "Learning Rate [float]:"));
    Settings["Num Epochs"]    = int(  Read_Setting(File, "Number of Epochs [int]:"));



    ############################################################################
    # Data settings.

    # Data file name. Note that the data file should NOT contain noise.
    Settings["DataSet Name"] =  Read_Setting(File, "DataSet [str]:");

    # All done! Return the settings!
    File.close();
    return Settings;
