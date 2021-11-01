import numpy;



def xy_Derivatives_To_Index(
            Num_x_Derivatives : int,
            Num_y_Derivatives : int) -> int:
    """ This function tells you which index value (in a multi-index)
    corresponds to each spatial partial derivative of U. You should only call
    this function if U is a function of TWO spatial variables.

    ----------------------------------------------------------------------------
    Arguments:

    Num_x_Derivatives, Num_y_Derivatives: The number of x, y derivatives in the
    spatial partial derivative of interest, respectively.

    For example, if we want to find the index value associated with the
    following spatial partial derivative of U:
            D_{x}^{i} D_{y}^{j} U
    Then Num_x_Derivatives = i, Num_y_Derivatives = j. Note that this function
    is a bijection from N^2 to N (actually, one can use this function to prove
    that N^2 and N have the same cardinality).

    ----------------------------------------------------------------------------
    Returns:

    An integer telling you which index value is associated with this particular
    spatial partial derivative. The table below gives a partial set of outcomes:
        (0, 0) -> 0
        (1, 0) -> 1
        (0, 1) -> 2
        (2, 0) -> 3
        (1, 1) -> 4
        (0, 2) -> 5
            ....
    """

    # Aliases.
    i : int = Num_x_Derivatives;
    j : int = Num_y_Derivatives;

    return ((i + j)*(i + j + 1))//2 + j;



def Index_to_xy_Derivatives_Array(Max_Derivatives : int):
    """ This function effectively generates a partial inverse to
    xy_Derivatives_To_Index. In particular, if Max_Derivatives = n, then it will
    generate a two column numpy array whose kth row specifies the input to
    xy_Derivatives_To_Index that maps to k.

    Note: You should ONLY use this function if working with 2 spatial variables.

    ----------------------------------------------------------------------------
    Arguments:

    Max_Derivatives: The maximum derivative order for which you want to invert
    xy_Derivatives_To_Index. This function will determine where the function
    sends all (i, j) in N^2 such that i + j <= Max_Derivatives. effectively,
    this value determines how much of xy_Derivatives_To_Index we find an
    inverse for. In general, this value should be the maximum number of
    derivatives of U you're taking (the setting value).

    ----------------------------------------------------------------------------
    Returns:

    A two column numpy array whose ith row holds the element of N^2 which
    xy_Derivatives_To_Index maps to i. In other words, it holds
    xy_Derivatives_To_Index^{-1}(i). """

    # alias.
    n : int = Max_Derivatives

    # The total number of spatial partial derivatives of order <= n is
    # 1 + 2 + ... n + (n + 1) = (n + 1)(n + 2)/2 (think about it). This is the
    # largest index value that we'll find an inverse for.

    # Set up return variable.
    Max_Index_Value : int = (n + 1)*(n + 2)//2;
    Inverse = numpy.empty((Max_Index_Value, 2));

    # Cycle through derivative order.
    i : int = 0;
    for k in range(0, n + 1):
        # Cycle through number of y derivatives.
        for j in range(0, k + 1):
            Inverse[i, 0] = k - j;      # Number of x derivatives.
            Inverse[i, 1] = j;          # Number of y derivatives.

            # Increment counter.
            i += 1;

    return Inverse;



def Num_Multi_Indices(
        num_sub_index_values : int,
        num_sub_indices      : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        counter              : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indices of
    specified num_sub_indices whose sub-indices take values in {0, 1...
    num_sub_index_values - 1}. Here, two multi-indices are "equal" if and only
    if there is a way to rearrange the sub-indices in one multi-index to match
    the others (both have the same value in each sub-index). This defines an
    equivalence relation of multi-indices. Thus, we are essentially finding the
    number of classes under this relation.

    For example, if num_sub_index_values = 4 and num_sub_indices = 2, then the
    set of possible multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus,
    in this case, this function would return 10.

    Note: we assume that num_sub_indices and num_sub_index_values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    num_sub_index_values: The number of distinct values that any one of the
    sub-indices can take on. If num_sub_index_values = k, then each sub-index
    can take on values 0, 1,... k-1.

    num_sub_indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    counter: stores the total number of multi-indices of specified oder whose
    sub-indices take values in 0, 1... num_sub_index_values - 1. We ultimately
    return this variable. It's passed as an argument for recursion.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indices (as defined above) which have
    num_sub_indices sub-indices, each of which takes values in {0, 1,...
    num_sub_index_values - 1}. """

    # Assertions.
    assert (num_sub_indices > 0), \
        ("num_sub_indices must be a POSITIVE integer. Got %d." % num_sub_indices);
    assert (num_sub_index_values > 0), \
        ("num_sub_index_values must be a POSITIVE integer. Got %d." % num_sub_index_values);

    # Base case
    if (sub_index == num_sub_indices - 1):
        return counter + (num_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            counter = Num_Multi_Indices(
                        num_sub_index_values = num_sub_index_values,
                        num_sub_indices      = num_sub_indices,
                        sub_index            = sub_index + 1,
                        sub_index_value      = j,
                        counter              = counter);

        return counter;



def Multi_Indices_Array(
        multi_indices        : numpy.array,
        num_sub_index_values : int,
        num_sub_indices      : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        position             : int = 0) -> int:
    """ This function finds the set of "distinct" multi-indices with
    num_sub_indices sub-indices such that each sub-index takes values in
    {0, 1,... num_sub_index_values - 1}. Here, two multi-indices are "equal" if
    and only if there is a way to rearrange the sub-indices in one multi-index
    to match the others (both have the same value in each sub-index). This
    defines an equivalence relation. Thus, we return a representative for each
    class.

    We assume that multi_indices is an N by num_sub_indices array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the num_sub_index_values and
    num_sub_indices arguments). This function populates the rows of
    multi_indices. The i,j element of multi_indices contains the value of the
    jth sub-index of the ith "distinct" (as defined above) multi-index.

    For example, if num_sub_index_values = 4 and num_sub_indices = 2, then the
    set of "distinct" multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }. This function will populate the
    first 10 rows of multi_indices as follows:
        [0, 0]
        [0, 1]
        [0, 2]
        [0, 3]
        [1, 1]
        [1, 2]
        [1, 3]
        [2, 2]
        [2, 3]
        [3, 3]

    Note: we assume that num_sub_indices and num_sub_index_values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    multi_indices: An N by num_sub_indices tensor, where N is "sufficiently
    large" (see above). This array will hold all distinct multi-indices with
    num_sub_indices sub-indices, each of which takes values in {0, 1,...
    num_sub_index_values - 1}.

    num_sub_index_values: The number of distinct values that any sub-index can
    take on. If num_sub_index_values = k, then each sub_index can take values
    0, 1,... num_sub_index_values - 1.

    num_sub_indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    ----------------------------------------------------------------------------
    Returns:

    An integer that the function uses for recursion. You probably want to
    discard it. """

    # Assertions.
    assert (num_sub_indices > 0), \
        ("num_sub_indices must be a POSITIVE integer. Got %d." % num_sub_indices);
    assert (num_sub_index_values > 0), \
        ("num_sub_index_values must be a POSITIVE integer. Got %d." % num_sub_index_values);

    # Base case
    if (sub_index == num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            multi_indices[position + (j - sub_index_value), sub_index] = j;

        return position + (num_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            new_position = Multi_Indices_Array(
                            multi_indices        = multi_indices,
                            num_sub_index_values = num_sub_index_values,
                            num_sub_indices      = num_sub_indices,
                            sub_index            = sub_index + 1,
                            sub_index_value      = j,
                            position             = position);

            for k in range(0, (new_position - position)):
                multi_indices[position + k, sub_index] = j;

            position = new_position;

        return position;



class Multi_Index_To_Col_Number():
    def __init__(   self,
                    Num_Sub_Index_Values : int,
                    Max_Sub_Indices      : int):

        self.Max_Sub_Indices      = Max_Sub_Indices;
        self.Num_Sub_Index_Values = Num_Sub_Index_Values;

        # First, allocate an array that's big enough to have a cell for every
        # possible multi-index. An array of size if M = Max_Sub_Indices, and
        # N = Num_Sub_Index_Values, then an array of size N^M is big enough for
        # this.
        M : int = Max_Sub_Indices;
        N : int = Num_Sub_Index_Values;
        self.Index_Array = numpy.empty(N**M, dtype = numpy.int64);

        # Now, populate the relevant elements of the Index_Array.
        Counter : int = 0;
        for k in range(1, Max_Sub_Indices + 1):
            # First, determine the number of multi-indices with k sub-indices,
            # each of which can take on Num_Sub_Index_Values values.
            Num_Multi_Indices_k : int = Num_Multi_Indices(
                                            num_sub_index_values = Num_Sub_Index_Values,
                                            num_sub_indices      = k);

            # Use this number to allocate an array to hold all the multi-indices
            # with k sub-indices, each of which can take on Num_Sub_Index_Values
            # values.
            Multi_Indices = numpy.empty((Num_Multi_Indices_k, k), dtype = numpy.int64);

            # Now use the Multi_Indices_Array function to populate that array.
            Multi_Indices_Array(
                multi_indices        = Multi_Indices,
                num_sub_index_values = Num_Sub_Index_Values,
                num_sub_indices      = k);

            # Now, for each multi-index, determine its index in the Index_Array,
            # populate that element with the current counter value.
            for j in range(0, Num_Multi_Indices_k):
                Multi_Index = Multi_Indices[j, :];
                Index = self.Multi_Index_To_Array_Index(Multi_Index);
                self.Index_Array[Index] = Counter;

                # Increment the counter.
                Counter += 1;



    def Multi_Index_To_Array_Index(Multi_Index : numpy.array) -> int:
        # First, get rid of any extra dimensions of size 1.
        Multi_Index = Multi_Index.squeeze();

        # Now, determine how many multi-indicies are in this multi-index.
        Num_Sub_Indices : int = Multi_Index.size;

        # Now, map that Multi-index to an array index according to the following
        # rule: If Max_Sub_Indices = M, Num_Sub_Index_Values = N, and
        # Num_Sub_Indices = K <= M, then,
        #       Index = Multi_Index[0]*N^{K - 1} + Multi_Index[1]*N^{K - 2} +.... Multi_Index[K]
        Index : int = 0;
        M : int = self.Num_Sub_Index_Values;
        for k in range(Num_Sub_Indices):
            Index += Multi_Index[k]*(M**(Num_Sub_Indices - 1 - k));

        return Index;


    def __call__(   self,
                    Multi_Index : numpy.array):
        """ This function maps a multi-index to its corresponding column
        number.

        ------------------------------------------------------------------------
        Arguments:

        Multi_Index: A 1 by N, or N by 1, or N element array that stores a
        multi-index with N sub-indices. We assume that N <= Max_Sub_Indices.
        We also assume that each sub-index takes values in {0, 1,...,
        Num_Sub_Index_Values}.

        ------------------------------------------------------------------------
        Returns:

        The column number associated with the Multi_Index. """

        # Remove any extra dimensions of size 1.
        Multi_Index = Multi_Index.squeeze();

        # Make sure Multi_Index has an appropiate number of sub-indices
        assert(Multi_Index.size <= self.Max_Sub_Indices);

        # Determine the index value associated with this multi index.
        Index : int = self.Multi_Index_To_Array_Index(Multi_Index);

        # Return the value in the corresponding cell of the Index_Array.
        return self.Index_Array(Index);
