import  numpy;
from    typing      import List;

from    Derivative  import Derivative;




class Term():
    """ We assume the system response function, u, satisfies a PDE of the
    following form:
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

    Powers : A list of integers. The kth entry of this list represents p(k) in
    the list above. """

    def __init__(   self,
                    Derivatives :   List[Derivative],
                    Powers      :   List[int]):
        """ This is the class initializer. If Derivatives and Powers have n
        elements, the ith entry of Derivatives corresponds to the partial
        derivative operator D^{(i)}, and the ith entry of Powers is p(i), then
        the initialized object represents the term
            (D^{(1)} u)^{p(1)} (D^{(2)} u)^(p(2)} ... (D^{(n)} u)^{p(n)}

        ------------------------------------------------------------------------
        Arguments:

        Derivatives : A list of derivative objects.

        Powers : A list of integers. This list should be the same length as
        Derivatives. """

        # Make sure Derivatives, Powers have the same length
        assert(len(Derivatives) == len(Powers));

        # Make sure each power is non-negative
        for i in range(len(Powers)):
            assert(Powers[i] >= 0);

        # Set up Derivatives, Powers.
        self.Derivatives    = Derivatives;
        self.Powers         = Powers;



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



    def __str__(self) -> str:
        """ This function returns a human-readable form of the term that self
        represents. """

        # Initialize Buffer.
        Buffer : str = "";

        # Cycle through the sub-terms.
        for i in range(len(self.Derivatives)):
            Buffer += "(" + self.Derivatives[i].__str__() + "U)^" + str(self.Powers[i]);


        # All done! Return!
        return Buffer;
