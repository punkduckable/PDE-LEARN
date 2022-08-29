import  torch;
from    typing  import Dict;

from File_Reader    import Read_Error, Read_Line_After, Read_Bool_Setting, Read_Setting, Read_Error;
from Library_Reader import Read_Library;



################################################################################
# Fucntions

def Index_After_Phrase(
        Line_In   : str,
        Phrase_In : str,
        Case_Sensitive : bool = False) -> int:
    """ This function searches for the substring Phrase_In within Line_In.

    ----------------------------------------------------------------------------
    Arguments:

    Line_In: A string. The program searches for the substring Phrase_In within
    Line_In.

    Phrase_In: A string containing the phrase we're searching for.

    Case_Sensitive: Controls if the search is case sensitive or not. (see
    Read_Line_After's docstring for more details).

    ----------------------------------------------------------------------------
    Returns:

    If Phrase_In is a substring of Line_In, then this returns the index of the
    first character after the first instance of Phrase_In within Line_In. If
    Phrase_In is NOT a substring of Line_In, then this function returns -1. """

    # First, get the number of characters in Line/Phrase.
    Num_Chars_Line   : int = len(Line_In);
    Num_Chars_Phrase : int = len(Phrase_In);

    # If we're ignoring case, then map Phrase, Line to lower case versions of
    # themselves. Note: We don't want to modify the original variables. Since
    # Python passes by references, we store this result in a copy of Line/Phrase
    Line   = Line_In;
    Phrase = Phrase_In;
    if(Case_Sensitive == False):
        Line = Line.lower();
        Phrase = Phrase.lower();

    # If Phrase is a substring of Line, then the first character of Phrase must
    # occur in the first (Num_Chars_Line - Num_Chars_Phrase) characters of
    # Line. Thus, we only need to check the first (Num_Chars_Line -
    # Num_Chars_Phrase) characters of Line.
    for i in range(0, Num_Chars_Line - Num_Chars_Phrase - 1):
        # Check if ith character of Line matches the 0 character of Phrase. If
        # so, check if  Line[i + j] == Phrase[j] for each for each j in {0,...
        # Num_Chars_Phrase - 1}.
        if(Line[i] == Phrase[0]):
            Match : Bool = True;

            for j in range(1, Num_Chars_Phrase):
                # If Line[i + j] != Phrase[j], then we do not have a match and
                # we should move onto the next character of Line.
                if(Line[i + j] != Phrase[j]):
                    Match = False;
                    break;

            # If Match is still True, then Phrase is a substring of Line and
            # i + Num_Chars_Phrase is the index of the first character in Line
            # after Phrase.
            if(Match == True):
                return i + Num_Chars_Phrase;

    # If we're here, then Phrase is NOT a substring of Line. Return -1 to
    # indiciate that.
    return -1;



def Read_Line_After(
        File,
        Phrase         : str,
        Comment_Char   : str = '#',
        Case_Sensitive : bool = False) -> str:
    """ This function tries to find a line of File that contains Phrase as a
    substring. Note that we start searching at the current position of the file
    pointer. We do not search from the start of File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to search for Phrase.

    Phrase: The Phrase we want to find. This should not include any instances
    of the comment char.

    Comment_Char: The character that denotes the start of a comment. If a line
    contains an instance of the Comment_Char, then we will remove everything in
    the line after the first instance of the Comment_Char before searching for
    a match.

    Case_Sensitive: Controls if the search is case sensitive or not. If
    True, then we search for an exact match (including case) of Phrase in one of
    File's lines. If not, then we try to find a line of File which contains the
    same letters in the same order as Phrase.

    ----------------------------------------------------------------------------
    Returns:

    If Phrase is a substring of a line of File, then this function returns
    everything in that line after the first occurrence of Phrase. If it can't
    find Phrase in one of File's lines, it raises an exception. If the Phrase is
    "cat is", and one of File's lines is "the cat is fat", then this will return
    " fat". """

    # Search the lines of File for one that contains Phrase as a substring.
    while(True):
        # Get the next line
        Line = File.readline();

        # Python doesn't use end of file characters. However, readline will
        # retun an empty string if and only if we're at the end of File. Thus,
        # we can use this as our "end of file" check
        if(Line == ""):
            raise Read_Error("Could not find \"" + Phrase + "\" in File.");

        # If the line is a comment, then ignore it.
        if (Line[0] == Comment_Char):
            continue;

        # Check for in-line comments.
        Line_Length = len(Line);
        for i in range(1, Line_Length):
            if(Line[i] == Comment_Char):
                Line = Line[:i-1];
                break;

        # Check if Phrase is a substring of Line. If so, this will return the
        # index of the first character after phrase in Line. In this case,
        # return everything in Line after that index. If Phrase is not in
        # Line. In this case, move on to the next line.
        Index : int = Index_After_Phrase(Line, Phrase, Case_Sensitive);
        if(Index == -1):
            continue;
        else:
            return Line[Index:];



def Read_Bool_Setting(File, Setting_Name : str) -> bool:
    """ Reads a boolean setting from File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to read the setting from.

    Setting_Name: The name of the setting we're reading. We need this in case
    of an error, so that we can print the appropiate error message.

    ----------------------------------------------------------------------------
    Return:

    The value of the boolean setting. """

    # Read the setting. This will yield a string.
    Buffer = Read_Line_After(File, Setting_Name).strip();

    # Check if the setting is present. If not, the Buffer will be empty.
    if  (len(Buffer) == 0):
        raise Read_Error("Missing Setting Value: You need to specify the \"%s\" setting" % Setting_Name);

    # Attempt to parse the result. Throw an error if we can't.
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        return True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        return False;
    else:
        raise Read_Error("Invalid Setting Value: \"%s\" should be \"True\" or \"False\". Got " % Setting_Name + Buffer);



def Read_Setting(File, Setting_Name : str) -> str:
    """ Reads a non-boolean setting from File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to read the setting from.

    Setting_Name: The name of the setting we're reading. We need this in case
    of an error, so that we can print the appropiate error message.

    ----------------------------------------------------------------------------
    Return:

    The value of the non-boolean setting as a string (you may need to type cast
    the returned value) """

    # Read the setting. This will yield a string.
    Buffer = Read_Line_After(File, Setting_Name).strip();

    # Check if the setting is present. If not, the Buffer will be empty.
    if  (len(Buffer) == 0):
        raise Read_Error("Missing Setting Value: You need to specify the \"%s\" setting" % Setting_Name);

    # Return!
    return Buffer;



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

    # Number of hidden layers in U network.
    Settings["Num Hidden Layers"]   = int(Read_Setting(File, "Number of Hidden Layers [int]:"));

    # Number of hidden units per hidden layer in the U network.
    Settings["Units Per Layer"]     = int(Read_Setting(File, "Hidden Units per Hidden Layer [int]:"));

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

    # Read Lambda value (used to scale the p-norm of Xi).
    Settings["Lambda"] = float(Read_Setting(File, "Lambda [float]:"));

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
