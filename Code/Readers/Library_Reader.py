# Nonsense to add Code directory to the Python search path.
import os
import sys

# Get path to parent directory (the Code directory, in this case)
Code_path       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
Classes_Path    = os.path.join(Code_path, "Classes");

# Add the Code directory to the python path.
sys.path.append(Classes_Path);

import  numpy;
from    typing          import List, Tuple, Dict;

from    File_Reader     import End_Of_File_Error, Read_Line_After, Read_Error;
from    Derivative      import Derivative, Get_Order;
from    Term            import Term;



def Parse_Sub_Term(Buffer : str) -> Tuple[Derivative, int]:
    """ 
    This function reads a Sub-Term; That is, an expression of the following 
    form:
            "(D_x^i D_y^j D_t^k u)^p"
    or
            "D_x^i D_y^j D_t^k u^p"
    In general, there may or may not be a power, p. Likewise, there may or may
    not be spaces between the derivative term and the '^', or between the '^' 
    and the power. Finally, there may or may not be any partial derivatives.

    ----------------------------------------------------------------------------
    Arguments:

    Buffer: This should be a string that contains a representation of the
    Library term we want to read. In general, this will be of the form
        (D_x^i D_y^j D_t^k u)^p.
    here, p must be a natural number. If there is no exponent, we infer p = 1. 
    """

    # First, split the term along the "u" or "U". This separates the partial 
    # derivative operator from the power (if there is one).    
    if('u' in Buffer):
        Components : List[str] = Buffer.split('u');
    else:
        Components : List[str] = Buffer.split('U');
    
    D_Component : str = Components[0].strip();
    p_Component : str = Components[1].strip();

    # The p_component is either empty, ")", of the form "^p", or of the 
    # form ")^p". In the first two cases, the power is 1. In the latter two, 
    # we can extract the power by splitting along the '^' character.
    Power : int = 1;
    if('^' in p_Component):
        Power = int(p_Component.split('^')[1].strip());

    # First, let's handle the case when there are no derivatives. In this case, 
    # the Sub-term is of the form U^p or (U)^p. In the former case, D_Component 
    # is empty, while in the latter it is "(". Further, there are the only 
    # cases where D_Component can have a length of less than 2. 
    if(len(D_Component) < 2):
        # In this case, we the derivative operator is just the identity.
        Encoding    : numpy.ndarray = numpy.zeros(shape = (2), dtype = numpy.int32);
        D           : Derivative    = Derivative(Encoding = Encoding);
       
        # And now we're done!
        return (D, Power);

    # The D_component should either be of the form "(D_x^i D_y^j D_t^k " or 
    # "D_x^i D_y^j D_t^k ". In the former case, we can just remove the '('.
    if(D_Component[0] == '('):
        D_Component = D_Component[1:];

    # At this point, the D_Component should be of the form "D_x^i D_y^j D_t^k".
    # It's time to split the derivatives into their components.
    D_Components : list[str] = D_Component.split(' ');

    # Each component should be of the form D_s^i or D_s, where i > 0, and s is
    # t, x, y, or z.
    Num_Derivative_Terms    : int           = len(D_Components);
    Encoding                : numpy.ndarray = numpy.zeros(shape = (4), dtype = numpy.int32);

    for i in range(Num_Derivative_Terms):
        # First, send everything to lower case (this makes processing easier).
        Component : str = D_Components[i].lower();

        # First, split at the '_'. The first character after this contains
        # the variable.
        Component : str = Component.split('_')[1];

        j : int = 0;
        if  (Component[0] == 't'):
            j = 0;
        elif(Component[0] == 'x'):
            j = 1;
        elif(Component[0] == 'y'):
            j = 2;
        elif(Component[0] == 'z'):
            j = 3;
        else:
            raise Read_Error("Derivative term has the wrong format. Buffer = "  + Buffer);

        # Each derivative term should either be of the form D_s or D_s^p. We
        # handle these cases separately.
        p : int = 1;
        if  ('^' in Component):
            p : int = int(Component[-1]);
        Encoding[j] += p;

    # Now, trim down the encoding (if possible).
    if(Encoding[3] == 0):
        Encoding = Encoding[0:3];

        if(Encoding[2] == 0):
            Encoding = Encoding[0:2];

    # We are now ready to build the derivative.
    D    : Derivative   = Derivative(Encoding = Encoding);

    # All done!
    return (D, Power);



def Parse_Term(Buffer : str) -> Term:
    """
    This function parses a term from a line of the Library file. The "Buffer"
    argument should be a stripped line of Library.txt that contains a term. In
    general, Read_Term is the only function that should call this one. Here, we
    read the term's sub-terms, build the term, and then return the resulting
    object.

    ----------------------------------------------------------------------------
    Arguments:

    Buffer: A stripped line containing a term. """

    # First, split the term into its sub-term using the * character.
    Sub_Terms : List[str] = Buffer.split("*");

    # Set up Derivatives, Powers lists.
    Derivatives = [];
    Powers      = [];

    # Now parse the sub-terms.
    for i in range(len(Sub_Terms)):
        # First, strip the sub-term.
        Sub_Term : str = Sub_Terms[i].strip();

        # Now parse it.
        Derivative, Power = Parse_Sub_Term(Sub_Term);

        # Append the results to the Derivatives, Powers lists.
        Derivatives.append(Derivative);
        Powers.append(Power);

    # Now, construct the Term object.
    return Term(Derivatives, Powers);



def Read_Term(File) -> Term:
    """ 
    This function reads a term (sequence of sub-terms, separated by *'s) from 
    the Library file. To do that, we search through the file for the first
    line that is neither blank nor entirely a comment. We then parse the term
    within.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to read a term from. This file should contain the
    terms (as strings) using the format specified in Library.txt. 
    """

    # Look for the next line that contains a library function.
    Line : str = "";
    while(True):
        # Get a candidate line. This eliminates all lines that start with a
        # comment or are blank. It will not, however, eliminate lines filled
        # with whitespace.
        Line = Read_Line_After( File    = File,
                                Phrase  = "");


        # Strip. If the line contains only whitespace, this will reduce it to
        # an empty string. If this is the case, move onto the next line.
        # Otherwise, the line should contain a library term.
        Line = Line.strip();
        if(len(Line) == 0):
            continue;
        else:
            break;

    # Now turn it into a Library Term object.
    return Parse_Term(Line);



def Read_Library(File_Path : str) -> Tuple[List[Derivative], Term, List[Term]]:
    """ 
    This function reads the Library terms in Library.txt.

    ----------------------------------------------------------------------------
    Arguments:

    File_Path: This is the name (relative to the working director of the file
    that houses the main function that called this one, and with a .txt
    extension of the library file). Thus, if we run this from Code/main.txt, and
    the Library function is Library.txt, then File_Path should be ../Library.txt
    """

    # First, open the file.
    File = open(File_Path, 'r');

    # Next, read the LHS Term. This is the first Library term in the file.
    LHS_Term : Term = Read_Term(File);

    # Finally, read the RHS Terms.
    RHS_Terms : List[Term] = [];
    while(True):
        try:
            Term : Term = Read_Term(File);
        except End_Of_File_Error:
            # If we raise this exception, then we're done.
            break;
        else:
            # Otherwise, add the new term to the RHS_Terms list.
            RHS_Terms.append(Term);

    # We can now close the file.
    File.close();


    ############################################################################
    # Now that we've read all the terms, lets make a list of the derivatives
    # we need, ordered by their order.

    # First, we must make a dictionary holding all the distinct derivatives
    # we need.
    Derivatives : Dict[Derivative] = {};

    # First, let's append the derivatives from the LHS term.
    for i in range(len(LHS_Term.Derivatives)):
        # Get the derivative.
        D : Derivative = LHS_Term.Derivatives[i];

        # Check if it's already in the dictionary.
        if(tuple(D.Encoding) in Derivatives):
            continue;

        # If not, add it to the dictionary.
        Derivatives[tuple(D.Encoding)] = D;

    # Next, repeat the above for each RHS term.
    for i in range(len(RHS_Terms)):
        for j in range(len(RHS_Terms[i].Derivatives)):
            # Get the derivative.
            D : Derivative = RHS_Terms[i].Derivatives[j];

            # Check if it's already in the dictionary.
            if(tuple(D.Encoding) in Derivatives):
                continue;

            # If not, add it to the dictionary.
            Derivatives[tuple(D.Encoding)] = D;

    # Now that we have the dictionary, we can convert it to a list and then
    # sort this list according to the order of the derivatives.
    Derivatives_List : List[Derivative] = list(Derivatives.values());

    # Sort them!
    Derivatives_List.sort(key = Get_Order);

    # All done!
    return Derivatives_List, LHS_Term, RHS_Terms;



def main():
    File_Path : str = "../../Library.txt";
    Derivatives, LHS_Term, RHS_Terms = Read_Library(File_Path = File_Path);

    # Print the Derivatives:
    print("Derivatives: ");
    print(Derivatives);

    # Print the terms.
    print("LHS Term = ", end = '');
    print(LHS_Term);

    # Print the RHS terms:
    print("RHS Terms: ");
    for i in range(len(RHS_Terms)):
        print(RHS_Terms[i]);


if __name__ == "__main__":
    main();
