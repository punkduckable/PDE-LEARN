# Nonsense to add Readers directory to the Python search path.
import os
import sys


# Get path to Reader directory.
Main_Path       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
Code_Path       = os.path.join(Main_Path, "Code");
Readers_Path    = os.path.join(Code_path, "Readers");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);

import torch;

from File_Reader    import Read_Line_After, Read_Error, Read_Bool_Setting, Read_Setting;



################################################################################
# Classes

class Settings_Container:
    # A container for data read in from the settings file.
    pass;



################################################################################
# Functions

def Settings_Reader() -> Settings_Container:
    """ This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Settings_Container object that contains all the settings we read from
    Settings.txt. The main function uses these to set up the program. """

    # Open file, initialze a Settings object.
    File        = open("Settings.txt", "r");
    Settings    = Settings_Container();

    # Where is the saved state?
    Settings.Load_File_Name     = Read_Setting(File, "Load File Name [str]:");

    # Number of hidden layers in U network.
    Settings.Num_Hidden_Layers  = int(Read_Setting(File, "Number of Hidden Layers [int]:"));


    ############################################################################
    # Network settings.

    # Number of hidden units per hidden layer in the U network.
    Settings.Units_Per_Layer    = int(Read_Setting(File, "Hidden Units per Hidden Layer [int]:"));

    # Which activation function should we use?
    Buffer = Read_Setting(File, "Activation Function [Tanh, Rational, Sin]:");
    if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
        Settings.Activation_Function = "Rational";
    elif(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Activation_Function = "Tanh";
    elif(Buffer[0] == 'S' or Buffer[0] == 's'):
        Settings.Activation_Function = "Sin";
    else:
        raise Read_Error("\"Activation Function [Tanh, Rational, Sin]:\" should be" + \
                         "\"Tanh\", \"Rational\", or \"Sin\" Got " + Buffer);


    ############################################################################
    # PDE settings.

    # What is the order of the time derivative in the PDE?
    Settings.Time_Derivative_Order      = int(Read_Setting(File, "Time Derivative Order [int]:"));

    # What is the highest order spatial derivative in the PDE?
    Settings.Max_Spatial_Derivatives    = int(Read_Setting(File, "Highest order spatial derivative of U [int]:"));

    # What is the maximum polynomial degree?
    Settings.Maximum_Term_Degree        = int(Read_Setting(File, "Maximum Polynomial Term Degree [int]:"));


    ############################################################################
    # DataSet settings.

    # Data file name. Note that the data file should NOT contain noise.
    Settings.Mat_File_Name =  Read_Setting(File, "Mat File Name [str]:");

    # All done! Return the settings!
    File.close();
    return Settings;