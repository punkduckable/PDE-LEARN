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



class Index_to_xy_Derivatives():
    """ This class effectively acts as a partial inverse to
    xy_Derivatives_To_Index. It does this using a look-up table whose kth
    row specifies the element of N^2 that xy_Derivatives_To_Index maps to k.

    Note: You should ONLY use this class if working with 2 spatial variables. """

    def __init__(self, Max_Derivatives : int):
        """ Max_Derivatives specifies the maximum derivative order for which you
        want to invert xy_Derivatives_To_Index. This function will determine
        where xy_Derivatives_To_Index sends each (i, j) in N^2 such that
                i + j <= Max_Derivatives.
        effectively, this value determines how much of xy_Derivatives_To_Index
        we find an inverse for. In general, this value should be the maximum
        number of derivatives of U you're taking (the setting value). We store
        the results in a lookup table, which the call method accesses. """

        # alias.
        n : int = Max_Derivatives;

        # The total number of spatial partial derivatives of order <= n is
        # 1 + 2 + ... n + (n + 1) = (n + 1)(n + 2)/2 (think about it). This is the
        # largest index value that we'll find an inverse for.

        # Set up look up table.
        self.Num_Index_Values : int = (n + 1)*(n + 2)//2;
        self.Lookup_Table = numpy.empty((self.Num_Index_Values, 2));

        # Cycle through derivative order.
        i : int = 0;
        for k in range(0, n + 1):
            # Cycle through the number of y derivatives.
            for j in range(0, k + 1):
                self.Lookup_Table[i, 0] = k - j; # Number of x derivatives.
                self.Lookup_Table[i, 1] = j;     # Number of y derivatives.

                # Increment counter.
                i += 1;

    def __call__(self, Index : int) -> numpy.array:
        """ This function returns the element of N^2 which
        xy_Derivatives_To_Index maps to Index.

        ------------------------------------------------------------------------
        Arguments:

        Index: the index value. This function determines which element of N^2
        xy_Derivatives_To_Index maps to Index.

        ------------------------------------------------------------------------
        Returns:

        A 1 by 2 numpy array. The first element specifies the number of x
        derivatives. the second specifies the number of y derivatives. Together,
        they give xy_Derivatives_To_Index^{-1}(Index). """

        assert(Index < self.Num_Index_Values);
        return self.Lookup_Table[Index, :];



def Num_Multi_Indices(
        Num_Sub_Indices      : int,
        Num_Sub_Index_Values : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        counter              : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indices of
    specified Num_Sub_Indices whose sub-indices take values in {0, 1...
    Num_Sub_Index_Values - 1}. Here, two multi-indices are "equal" if and only
    if there is a way to rearrange the sub-indices in one multi-index to match
    the others (both have the same value in each sub-index). This defines an
    equivalence relation of multi-indices. Thus, we are essentially finding the
    number of classes under this relation.

    For example, if Num_Sub_Index_Values = 4 and Num_Sub_Indices = 2, then the
    set of possible multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus,
    in this case, this function would return 10.

    Note: we assume that Num_Sub_Indices and Num_Sub_Index_Values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    Num_Sub_Index_Values: The number of distinct values that any one of the
    sub-indices can take on. If Num_Sub_Index_Values = k, then each sub-index
    can take on values 0, 1,... k-1.

    Num_Sub_Indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    counter: stores the total number of multi-indices of specified oder whose
    sub-indices take values in 0, 1... Num_Sub_Index_Values - 1. We ultimately
    return this variable. It's passed as an argument for recursion.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indices (as defined above) which have
    Num_Sub_Indices sub-indices, each of which takes values in {0, 1,...
    Num_Sub_Index_Values - 1}. """

    # Assertions.
    assert (Num_Sub_Indices > 0), \
        ("Num_Sub_Indices must be a POSITIVE integer. Got %d." % Num_Sub_Indices);
    assert (Num_Sub_Index_Values > 0), \
        ("Num_Sub_Index_Values must be a POSITIVE integer. Got %d." % Num_Sub_Index_Values);

    # Base case
    if (sub_index == Num_Sub_Indices - 1):
        return counter + (Num_Sub_Index_Values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < Num_Sub_Indices - 1):
        for j in range(sub_index_value, Num_Sub_Index_Values):
            counter = Num_Multi_Indices(
                        Num_Sub_Indices      = Num_Sub_Indices,
                        Num_Sub_Index_Values = Num_Sub_Index_Values,
                        sub_index            = sub_index + 1,
                        sub_index_value      = j,
                        counter              = counter);

        return counter;



def Multi_Indices_Array(
        Multi_Indices        : numpy.array,
        Num_Sub_Indices      : int,
        Num_Sub_Index_Values : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        position             : int = 0) -> int:
    """ This function finds the set of "distinct" multi-indices with
    Num_Sub_Indices sub-indices such that each sub-index takes values in
    {0, 1,... Num_Sub_Index_Values - 1}. Here, two multi-indices are "equal" if
    and only if there is a way to rearrange the sub-indices in one multi-index
    to match the others (both have the same value in each sub-index). This
    defines an equivalence relation. Thus, we return a representative for each
    class.

    We assume that Multi_Indices is an N by Num_Sub_Indices array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the Num_Sub_Index_Values and
    Num_Sub_Indices arguments). This function populates the rows of
    Multi_Indices. The i,j element of Multi_Indices contains the value of the
    jth sub-index of the ith "distinct" (as defined above) multi-index.

    For example, if Num_Sub_Index_Values = 4 and Num_Sub_Indices = 2, then the
    set of "distinct" multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }. This function will populate the
    first 10 rows of Multi_Indices as follows:
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

    Note: we assume that Num_Sub_Indices and Num_Sub_Index_Values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    Multi_Indices: An N by Num_Sub_Indices tensor, where N is "sufficiently
    large" (see above). This array will hold all distinct multi-indices with
    Num_Sub_Indices sub-indices, each of which takes values in {0, 1,...
    Num_Sub_Index_Values - 1}.

    Num_Sub_Index_Values: The number of distinct values that any sub-index can
    take on. If Num_Sub_Index_Values = k, then each sub_index can take values
    0, 1,... Num_Sub_Index_Values - 1.

    Num_Sub_Indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    ----------------------------------------------------------------------------
    Returns:

    An integer that the function uses for recursion. You probably want to
    discard it. """

    # Assertions.
    assert (Num_Sub_Indices > 0), \
        ("Num_Sub_Indices must be a POSITIVE integer. Got %d." % Num_Sub_Indices);
    assert (Num_Sub_Index_Values > 0), \
        ("Num_Sub_Index_Values must be a POSITIVE integer. Got %d." % Num_Sub_Index_Values);

    # Base case
    if (sub_index == Num_Sub_Indices - 1):
        for j in range(sub_index_value, Num_Sub_Index_Values):
            Multi_Indices[position + (j - sub_index_value), sub_index] = j;

        return position + (Num_Sub_Index_Values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < Num_Sub_Indices - 1):
        for j in range(sub_index_value, Num_Sub_Index_Values):
            new_position = Multi_Indices_Array(
                            Multi_Indices        = Multi_Indices,
                            Num_Sub_Indices      = Num_Sub_Indices,
                            Num_Sub_Index_Values = Num_Sub_Index_Values,
                            sub_index            = sub_index + 1,
                            sub_index_value      = j,
                            position             = position);

            for k in range(0, (new_position - position)):
                Multi_Indices[position + k, sub_index] = j;

            position = new_position;

        return position;



class Multi_Index_To_Col_Number():
    """ This class creates a functor that maps multi-indicies to their
    corresponding column number (in the library). In particular, given a maximum
    number of sub-indicies, as well as the number of values that each sub-index
    can take on, this class generates an injective function which maps those
    multi-indices to {0, 1,... K-1}, where K is the number of multi-indicies
    that meet the criteria. """
    def __init__(   self,
                    Max_Sub_Indices      : int,
                    Num_Sub_Index_Values : int):

        """" For brevity, met M = Max_Sub_Indices and N = Num_Sub_Index_Values.
        This function essentially generates a pair of look-up tables which,
        when taken together, specify which column number is associated with each
        multi-index that has at most M sub-indicies, each of which takes on
        N possible values.

        The tricky part is that we need to associate a unique number with each
        Multi-Index. Ideally, we'd like this to just be the column number.
        Unfortuneatly, there's no good way to do that directly. So, I've taken
        a different approach: Map the multi-index [i(1), i(2),... i(m)] to
            (i(1) + 1)*(N^(m - 1)) + (i(2) + 1)*(N^(m - 2)) + ... + (i(m) + 1) - 1
        (the -1 at the end is for zero indexing). This mapping is injective,
        but not surjective. In fact, it doesn't map to most values in N. We
        create an array that is big enough to hold the range of this map when
        m <= M and each i(k) takes values in {0, 1, ..., N - 1}. We then assign
        a column  number to each multi-index, [i(1), ... i(m)], and then store
        that number in ith cell of the array, where i is the image of [i(1),...
        i(m)] under the above map.

        The underlying array can be gigantic, and most of it goes unused. We
        could use a sparse representation, but we need a matrix that has O(1)
        access time. Since the array does not have any clear structure (to me,
        at least), I'm not sure how to do that. So, this functor is not
        terribly space efficient, but it works. """

        self.Max_Sub_Indices      = Max_Sub_Indices;
        self.Num_Sub_Index_Values = Num_Sub_Index_Values;

        # First, allocate an array that's big enough to have a cell for every
        # possible multi-index. An array of size if M = Max_Sub_Indices, and
        # N = Num_Sub_Index_Values, then an array of size N^M + N^(M - 1) +...
        # N is big enough for this.
        M : int = Max_Sub_Indices;
        N : int = Num_Sub_Index_Values;
        Size = 0;
        for k in range(1, M + 1):
            Size += N**k;

        self.Index_Array = numpy.empty(Size, dtype = numpy.int64);

        # Now, populate the relevant elements of the Index_Array.
        Counter : int = 0;
        for k in range(1, Max_Sub_Indices + 1):
            # First, determine the number of multi-indices with k sub-indices,
            # each of which can take on Num_Sub_Index_Values values.
            Num_Multi_Indices_k : int = Num_Multi_Indices(
                                            Num_Sub_Indices      = k,
                                            Num_Sub_Index_Values = Num_Sub_Index_Values);

            # Use this number to allocate an array to hold all the multi-indices
            # with k sub-indices, each of which can take on Num_Sub_Index_Values
            # values.
            k_Indices = numpy.empty((Num_Multi_Indices_k, k), dtype = numpy.int64);

            # Now use the Multi_Indices_Array function to populate that array.
            Multi_Indices_Array(
                Multi_Indices        = k_Indices,
                Num_Sub_Indices      = k,
                Num_Sub_Index_Values = Num_Sub_Index_Values);

            # Now, for each multi-index, determine its index in the Index_Array,
            # populate that element with the current counter value.
            for j in range(0, Num_Multi_Indices_k):
                Multi_Index = k_Indices[j, :];
                Index = self.Multi_Index_To_Array_Index(Multi_Index);
                self.Index_Array[Index] = Counter;

                # Increment the counter.
                Counter += 1;



    def Multi_Index_To_Array_Index(self, Multi_Index : numpy.array) -> int:
        # First, get rid of any extra dimensions of size 1.
        Multi_Index = Multi_Index.reshape(-1);

        # Now, determine how many multi-indicies are in this multi-index.
        Num_Sub_Indices : int = Multi_Index.size;

        # Now, map that Multi-index to an array index according to the following
        # rule: If Max_Sub_Indices = M, Num_Sub_Index_Values = N, and
        # Num_Sub_Indices = K <= M, then,
        #       Index = (Multi_Index[0] + 1)*N^{K - 1}
        #             + (Multi_Index[1] + 1)*N^{K - 2}
        #             ...
        #             + (Multi_Index[K] + 1)
        #             - 1
        # Note that the -1 is here for zero-indexing.
        Index = 0;
        M : int = self.Num_Sub_Index_Values;
        for k in range(0, Num_Sub_Indices):
            Index += (Multi_Index[k] + 1)*(M**(Num_Sub_Indices - 1 - k));
        Index -= 1;

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
        Multi_Index = Multi_Index.reshape(-1);

        # Make sure Multi_Index has an appropiate number of sub-indices
        assert(Multi_Index.size <= self.Max_Sub_Indices);

        # Determine the index value associated with this multi index.
        Index : int = self.Multi_Index_To_Array_Index(Multi_Index);

        # Return the value in the corresponding cell of the Index_Array.
        return self.Index_Array[Index];



class Col_Number_To_Multi_Index():
    """ This class creates functors which map column numbers back to their
    corresponding multi-indicies. Specicially, its object acts as an inverse to
    the functors created by Multi_Index_To_Col_Number. It accomplishes this
    using a pair of look-up tables.

    Suppose that f is an instance of Multi_Index_To_Col_Number that was
    initialized with Max_Sub_Indices = M, and Num_Sub_Index_Values = N. Let
    g be an instance of Col_Number_To_Multi_Index that was also initialized with
    Max_Sub_Indices = M and Num_Sub_Index_Values = N. Let's focus on g's
    look up table: The kth row of this table specifies two valies, i and j.
    i specifies the number of sub-indices in the multi-index that f maps
    to k. g also contains an array which holds all multi-indicies that have
    i sub-indices. j tells us which element of that array holds the multi-index
    that f mapped to k. Thus, once we have i and j, we can just return the
    jth element of the array that holds the multi-indicies with i sub-indices.

    In this way, our look-up table basically tells us which other look-up table
    to go to, as well as which element of that table holds the answer we want. """

    def __init__(   self,
                    Max_Sub_Indices      : int,
                    Num_Sub_Index_Values : int):

        # First, allocate a list of arrays holding all possible multi-indices.
        # Keep track of how many total indices there are.
        self.Multi_Indices = [];
        self.Total_Indices : int = 0;
        for k in range(1, Max_Sub_Indices + 1):
            # Determine how many multi-indices there are with k sub-indices,
            # each of which can take on Num_Sub_Index_Values values.
            Num_k_Indices : int = Num_Multi_Indices(
                                        Num_Sub_Indices      = k,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);
            self.Total_Indices += Num_k_Indices;

            k_Indices = numpy.empty((Num_k_Indices, k), dtype = numpy.int64);
            Multi_Indices_Array(
                    Multi_Indices        = k_Indices,
                    Num_Sub_Indices      = k,
                    Num_Sub_Index_Values = Num_Sub_Index_Values);

            self.Multi_Indices.append(k_Indices);

        # The idea is as follows: We'll make a look-up table whose ith row
        # has two elements. The first tells you how many sub-indices are in
        # the multi-index associated with the ith column. The second will tell
        # you which element of self.Multi_Indicies[k] holds the actual
        # multi-index. First, we'll allocate the lookup table:
        self.Lookup_Table = numpy.empty((self.Total_Indices, 2), dtype = numpy.int64);

        # Now, make a Multi_Index_To_Col_Number map. We'll use this to populate
        # the Lookup table. This way, our function will literally be an
        # inverse of Multi_Index_To_Col_Number. Even if the way that we map the
        # multi-indicies to columns changes, this function will still work.
        # (as long as Multi_Index_To_Col_Number maps N possible multi-indicies
        # to 0, 1,... N-1)
        Index_To_Col = Multi_Index_To_Col_Number(
                            Max_Sub_Indices      = Max_Sub_Indices,
                            Num_Sub_Index_Values = Num_Sub_Index_Values);

        # Populate the lookup table!
        for k in range(0, Max_Sub_Indices):
            Num_kp1_Indices : int = self.Multi_Indices[k].shape[0];
            for j in range(Num_kp1_Indices):
                Multi_Index   = self.Multi_Indices[k][j, :];
                Col_Num : int = Index_To_Col(Multi_Index);

                self.Lookup_Table[Col_Num, 0] = k;
                self.Lookup_Table[Col_Num, 1] = j;

    def __call__(self, Col_Num : int):
        """ This function returns the multi-index corresponding to the specified
        Column nuber.

        ------------------------------------------------------------------------
        Arguments:

        Col_Number: The column number whose corresponding multi-index we want.
        Specifcially, suppose that Max_Sub_Indices = M and
        Num_Sub_Index_Values = N. Let f be an instance of
        Multi_Index_To_Col_Number that was initialized using N and M. Then,
        this function with Col_Num = k gives the multi-index that f maps to k.

        ------------------------------------------------------------------------
        Returns:

        The multi-index (as a numpy array row) corresponding to the Col_Num. See
        description under Col_Number in the Arguments section of this doc
        string. """

        # First, get the number of sub-indices in the multi-index corresponding
        # to this column number, as well as the cell of the corresponding
        # array in self.Multi_Indices holds the corresponding multi-index.
        k : int = self.Lookup_Table[Col_Num, 0];
        j : int = self.Lookup_Table[Col_Num, 1];

        assert(Col_Num < self.Total_Indices);

        return self.Multi_Indices[k][j, :];
