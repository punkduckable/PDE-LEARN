import numpy;



class Derivative():
    """ 
    Objects of this class house an abstract representation of a partial 
    derivative operator.

    ----------------------------------------------------------------------------
    Members:

    Encoding : A 1D numpy array of integers characterizing the partial
    derivative operator. If there are n spatial variables, then this should be a
    n + 1 element array, whose 0 element holds the number of time derivatives,
    and whose ith element (for i > 0) holds the derivative order with respect to
    the i-1th spatial variable. Currently, we only support n = 1, 2, 3.

    Order : This is the sum of the elements of Encoding. It represents the
    total number of partial derivatives we must take to apply a derivative
    operator to a function. We need this when computing the integral of a weight
    function times a library term (we apply integration by parts once for each
    partial derivative in the Derivative operator).
    """


    def __init__(   self,
                    Encoding : numpy.ndarray) -> None:
        """ 
        Initializer.

        ------------------------------------------------------------------------
        Arguments:

        Encoding: See class docstring. 
        """

        # First, cast to integer array. This also returns a copy of Encoding.
        Encoding : numpy.ndarray = Encoding.astype(dtype = numpy.int32);

        # check that Encoding is a 1D array.
        assert(len(Encoding.shape) == 1);

        # Determine input dimension. Currently, we only support n = 2, 3, 4.
        Input_Dim : int  = Encoding.size;
        assert(Input_Dim == 2 or Input_Dim == 3 or Input_Dim == 4);

        # Calculate the order (total number of partial derivatives) of this
        # derivative operator.
        self.Order : int = numpy.sum(Encoding).item();

        # Check that each element of encoding is a non-negative integer.
        for i in range(Encoding.size):
            assert(Encoding[i] >= 0);

        # Assuming the input passes all checks, assign it.
        self.Encoding = Encoding;



    def __str__(self) -> str:
        """ 
        This function returns a string that contains a human-readable
        expression for the derivative operator that this object represents. It
        is mainly used for printing. 
        """

        Buffer : str = "";

        # Time derivative.
        if  (self.Encoding[0] == 1):
            Buffer += "D_t ";
        elif(self.Encoding[0] > 1 ):
            Buffer += ("D_t^%u " % self.Encoding[0]);

        # x derivative.
        if  (self.Encoding[1] == 1):
            Buffer += "D_x ";
        elif(self.Encoding[1] > 1 ):
            Buffer += ("D_x^%u " % self.Encoding[1]);

        # y derivative (if it exists)
        Num_Spatial_Vars : int = self.Encoding.size - 1;
        if(Num_Spatial_Vars > 1):
            if  (self.Encoding[2] == 1):
                Buffer += "D_y ";
            elif(self.Encoding[2] > 1 ):
                Buffer += ("D_y^%u " % self.Encoding[2]);

        # z derivative (if it exists).
        if(Num_Spatial_Vars > 2):
            if  (self.Encoding[3] == 1):
                Buffer += "D_z ";
            elif(self.Encoding[3] > 1 ):
                Buffer += ("D_z^%u " % self.Encoding[3]);

        return Buffer;



    def Is_Child_Of(self, D) -> bool:
        """ 
        This function determines if D is a "child" of the derivative operator 
        D. What does this mean? Let D_1 and D_2 be derivative objects.
        Let [p_1, ... , p_n] and [q_1, ... , q_m] denote D_1's and D_2's
        encoding vectors, respectively. Let l = min{n, m}. D_1 is a child of
        D_2 if and only if for k \in {1, 2, ... , l}, p_k <= q_k and p_k for
        k > l. This is equivalent to saying that there is a derivative operator
        D_3 such that D_2 U = D_3 D_1 U.

        ------------------------------------------------------------------------
        Arguments:

        D : A derivative operator. This function determines if self is a child
        of D.


        ------------------------------------------------------------------------
        Returns:

        A boolean; True if self is a child of D, False otherwise. 
        """

        # First, find l, the minimum length of self.Encoding and D.Encoding.
        n : int = len(self.Encoding);
        m : int = len(D.Encoding);
        l : int = min(n, m);

        # Check that self.Encoding[k] <= D.Encoding[k] for k <= l.
        for k in range(l):
            if(self.Encoding[k] > D.Encoding[k]):
                return False;

        # Assuming self.Encoding is longer than D.Encoding, make sure that the
        # components of self.Encoding after the l'th are zero.
        for k in range(l + 1, n):
            if(self.Encoding[k] != 0):
                return False;

        # If we make it here, then self is a child of D.
        return True;



def Get_Order(D : Derivative):
    """ 
    This function returns D's order. This function exists only to enable
    sorting with lists of derivative operators. 
    """

    return D.Order;
