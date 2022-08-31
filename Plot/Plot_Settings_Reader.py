# Nonsense to add Readers directory to the Python search path.
import os
import sys

# Get path to Reader directory.
Main_Path       : str = os.path.dirname(os.path.abspath(os.path.curdir));
Code_Path       : str = os.path.join(Main_Path, "Code");
Readers_Path    : str = os.path.join(Code_Path, "Readers");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);

import  torch;
from    typing          import Dict, List;

from    File_Reader     import Read_Line_After, Read_Error, Read_Bool_Setting, Read_Setting;
from    Library_Reader  import Read_Library;



def Settings_Reader() -> Dict:
    """ 
    This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Dictionary housing the settings we read from Settings.txt. The main 
    function uses these to set up the program. 
    """

    # Open file, initialize a Settings object.
    File        = open("Settings.txt", "r");
    Settings    = {};

    # Where is the saved state?
    Settings["Load File Name"]  = Read_Setting(File, "Load File Name [str]:");

    # Data file name. Note that the data file should NOT contain noise.
    Settings["Mat File Name"]   = Read_Setting(File, "Mat File Name [str]:");

    # All done! Return the settings!
    File.close();
    return Settings;
