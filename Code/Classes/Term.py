import  numpy;
from    typing      import List, Dict;

from    Derivative  import Derivative;




class Term():
    """ 
    We assume the system response function, u, satisfies a PDE of the following
    form:
            T_1(U) = \sum_{k = 1}^{N} c_k T_k(U)
    where each T_k is an expression of the form
            T_k(u) = (D^{(1)} u)^{p(1)} (D^{(2)} u)^(p(2)} ... (D^{(n)} u)^{p(n)}
    here, each D^{(k)} is a partial derivative operator and p(k) is a
    non-negative integer. We call each function T_k a "term". The term class
    acts as an abstract representation of a term. For brevity, we also refer to
    an expression of the form (D^{(k)} u)^{p(k)} as a "sub-term".

    ----------------------------------------------------------------------------
    Members:

    Derivatives : A list of Derivative objects. The kth entry of this list
    represents D^{(k)} in the expression above.

    Powers : A list of natural numbers. The kth entry of this list represents
    p(k) in the list above.

    Num_Sub_Terms : The number of sub terms in the Term. Equivalently, the
    length of Derivatives and Powers.
    """



    def __init__(   self,
                    Derivatives :   List[Derivative],
                    Powers      :   List[int]):
        """ 
        This is the class initializer. If Derivatives and Powers have n
        elements, the ith entry of Derivatives corresponds to the partial
        derivative operator D^{(i)}, and the ith entry of Powers is p(i), then
        the initialized object represents the term
            (D^{(1)} u)^{p(1)} (D^{(2)} u)^(p(2)} ... (D^{(n)} u)^{p(n)}

        ------------------------------------------------------------------------
        Arguments:

        Derivatives : A list of derivative objects.

        Powers : A list of natural numbers. This list should be the same length
        as Derivatives. 
        """

        # Make sure Derivatives, Powers have the same length
        assert(len(Derivatives) == len(Powers));

        # Make sure each power is positive
        for i in range(len(Powers)):
            assert(Powers[i] >= 1);

        # Set up Derivatives, Powers.
        self.Derivatives    = Derivatives;
        self.Powers         = Powers;
        self.Num_Sub_Terms  = len(Powers);



    def Append( self,
                Derivative      : Derivative,
                Power           : int) -> None:
        """ This function appends a new Derivative, power pair to the term. If
        Derivative represents the partial derivative operator D, Power is p, and
        self currently represents the term
                (D^{(1)} u)^{p(1)} ... (D^{(n)} u)^{p(n)},
        then this function modifies self to represent the following term
                (D^{(1)} u)^{p(1)} ... (D^{(n)} u)^{p(n)} (D u)^p

        ------------------------------------------------------------------------
        Arguments:

        Derivatives : A derivative object. """

        # Make sure Power >= 0.
        assert(Power >= 0);

        # Append new entries to the Derivatives, Powers lists.
        self.Derivatives.append(Derivative);
        self.Powers.append(Power);
        self.Num_Sub_Terms += 1;



    def Get_State(self) -> Dict:
        """
        This function helps serialize self. It returns a dictionary that can be
        used to create self from scratch. You can recover a copy of self by 
        passing this dictionary to the Build_Term_From_State function.
        
        -----------------------------------------------------------------------
        Returns:

        A dictionary with two keys, "Derivative Encodings" and "Powers". The 
        former is a list whose ith entry holds the Encoding array for the ith 
        derivative of the library term. The latter is simply self's power 
        attribute.
        """

        # We can start things off with the Powers attribute.
        State : Dict = {"Powers" : self.Powers};

        # We can now build the derivatives attribute.
        Encodings : List[numpy.ndarray] = [];
        for i in range(len(self.Derivatives)):
            Encodings.append(self.Derivatives[i].Encoding);
        
        State["Derivative Encodings"] = Encodings;

        return State;



    def __str__(self) -> str:
        """ This function returns a human-readable form of the term that self
        represents. """

        # Initialize Buffer.
        Buffer : str = "";

        # Cycle through the sub-terms, if there are any.
        if(len(self.Derivatives) >= 1):
            for i in range(len(self.Derivatives)):
                # Append the sub-terms's derivative.
                Buffer += "(" + self.Derivatives[i].__str__() + "U)";

                # Append the sub-term's power if it's > 1.
                if(self.Powers[i] > 1):
                    Buffer += "^" + str(self.Powers[i]);

                # Append the '*' symbol between sub-terms.
                if(i != len(self.Derivatives) - 1):
                    Buffer += "*";

        # All done! Return!
        return Buffer;



def Build_Term_From_State(State : Dict) -> Term:
    """
    This function builds a new term object from a State dictionary. It then
    returns that object. 

    ---------------------------------------------------------------------------
    Arguments:

    State: A dictionary. This should either be the dictionary returned by the 
    Get_State method, or an unpickled copy of one. 

    ---------------------------------------------------------------------------
    Returns:

    A new Term object whose derivatives and powers are specified in State.
    """

    # To build a State object, we need to build the "Powers"and "Derivatives" 
    # variables. State contains both, but the latter is stored as a list of 
    # Encoding vectors (we need to build the actual derivatives ourselves)
    Powers          : List[int]         = State["Powers"];
    Derivatives     : List[Derivative]  = [];

    Num_Derivatives = len(State["Derivative Encodings"]);
    for i in range(Num_Derivatives):
        # build the ith derivative, append it to the List.
        Derivatives.append(Derivative(Encoding = State["Derivative Encodings"][i]));
    
    # We can now build the Term object.
    return Term(Derivatives = Derivatives, Powers = Powers);