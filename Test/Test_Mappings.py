# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# Now we can do our usual import stuff.
import numpy;
import torch;
import unittest;
import random;
import math;

# Code files.
from Mappings import xy_Derivatives_to_Index, Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, \
                     Num_Multi_Indices, Multi_Indices_Array, \
                     Multi_Index_to_Col_Number_Class, Col_Number_to_Multi_Index_Class;



class xy_Derivatives_And_Index(unittest.TestCase):
    def test_xy_Derivatives_to_Index(self):
        # Test that the function gives the expected output for derivatives less
        # than a specified order, M. Note that we do not care about the order in
        # which xy_Derivatives_to_Index maps elements of N^2 to N, so long as
        # all elements of N^2 whose elements sum to <= M are mapped to {0, 1,
        # ... (M + 1)(M + 2)/2 - 1}.

        M : int = random.randrange(1, 10);
        Num_Index_Values : int = ((M + 1)*(M + 2))//2;

        # To track how many pairs get mapped to each element in {0, 1,...
        # (M + 1)(M + 2)/2 - 1}
        Hit_Counter = numpy.zeros(Num_Index_Values, dtype = numpy.int64);

        for k in range(0, M + 1):
            # There are k+1 spatial partial derivatives of order k. They are
            #       D_{x}^k U, D_{x}^{k - 1} D_{y} U,... D_{y}^{k} U.
            for j in range(0, k + 1):
                # Number of x derivatives is k - j.
                i     : int = k - j;
                Index : int = xy_Derivatives_to_Index(i, j);

                # check that the Index is in {0, 1,... (M + 1)(M + 2)/2 - 1}.
                self.assertGreaterEqual(Index, 0);
                self.assertLess(Index, Num_Index_Values);

                Hit_Counter[Index] += 1;

                #print("(%d, %d) -> %d" % (i, j, Index));

        # now check that each value in {0, 1,... (M + 1)(M + 2)/2} got hit once.
        for i in range(Num_Index_Values):
            self.assertEqual(Hit_Counter[i], 1);



    def test_Index_to_xy_Derivatives_Array(self):
        # Generate a partial inverse, then test that this inverse actually
        # works.

        # First, pick a random index (with a reasonable value).
        Highest_Order_Derivatives : int = random.randrange(1, 10);

        # Now generate the inverse list.
        Index_to_xy_Derivatives = Index_to_xy_Derivatives_Class(Highest_Order_Derivatives);

        # Determine maximum index value that we inverted.
        N = Index_to_xy_Derivatives.Num_Index_Values;

        # Check that Index_to_xy_Derivatives composed with
        # xy_Derivatives_to_Index is the identity.
        for k in range(N):
            xy_Deriv = Index_to_xy_Derivatives(Index = k);
            i : int = xy_Deriv[0];
            j : int = xy_Deriv[1];
            Index_Num : int = xy_Derivatives_to_Index(i, j);
            self.assertEqual(k, Index_Num);

            #print("%d -> (%d, %d) -> %d" % (k, i, j, Index_Num));

        # check that xy_Derivatives_to_Index composed with
        # Index_to_xy_Derivatives is the identity.
        for k in range(0, Highest_Order_Derivatives):
            for j in range(0, k + 1):
                i : int = k - j;        # Number of x derivatives.

                Index : int = xy_Derivatives_to_Index(i, j);
                xy_Der = Index_to_xy_Derivatives(Index).reshape(-1);

                self.assertEqual(xy_Der[0].item(), i);
                self.assertEqual(xy_Der[1].item(), j);

                #print("(%d, %d) -> %d -> (%d, %d)" % (i, j, Index, xy_Der[0].item(), xy_Der[1].item()));



class Multi_Index_And_Col_Number(unittest.TestCase):
    def test_Multi_Index_to_Col_Number(self):
        # First, determine how many sub-indices we want to use, and how many
        # index values we want per sub-index.
        Max_Sub_Indices      : int = random.randrange(1, 6);
        Num_Sub_Index_Values : int = random.randrange(2, 15);

        Multi_Index_to_Col_Number = Multi_Index_to_Col_Number_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        # Now, generate the set of all possible multi-indices with at most
        # Max_Sub_Indices sub-indices, each of which can take on
        # Num_Sub_Index_Values values.
        Multi_Indices = [];
        Total_Indices : int = 0;
        for k in range(1, Max_Sub_Indices + 1):
            # Determine the number of multi-indicies with k sub-indicies, each
            # of which can take on Num_Sub_Index_Values values.
            Num_k_Indices : int = Num_Multi_Indices(
                                        Num_Sub_Indices      = k,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);
            Total_Indices += Num_k_Indices;

            # Acquire a list of all possible multi-indicies with k sub-indices,
            # each of wich can take on Num_Sub_Index_Values values.
            Multi_Indices_k = numpy.empty((Num_k_Indices, k), dtype = numpy.int64);
            Multi_Indices_Array(
                Multi_Indices        = Multi_Indices_k,
                Num_Sub_Indices      = k,
                Num_Sub_Index_Values = Num_Sub_Index_Values);

            # Append to the master list.
            Multi_Indices.append(Multi_Indices_k);

        # Now, see where each Multi_Index gets mapped to.
        Counter = 0;
        Hit_Counter = numpy.zeros(Total_Indices, dtype = numpy.int64);
        for k in range(0, Max_Sub_Indices):
            Num_k_Indices = Multi_Indices[k].shape[0];

            for j in range(0, Num_k_Indices):
                # Each multi-index should be mapped to a value in {0, 1,... Total_Indices - 1}.
                Col_Num : int = Multi_Index_to_Col_Number(Multi_Indices[k][j, :]);
                self.assertLess(Col_Num, Total_Indices);
                Hit_Counter[Col_Num] += 1;

                #Multi_Index = Multi_Indices[k][j, :];
                #print("[ ", end = '');
                #for i in range(0, k + 1):
                #    print("%2d " % Multi_Index[i], end = '');
                #print("] -> %d" % Multi_Index_to_Col_Number(Multi_Index));

        # Check that each element of the range is hit exactly once.
        for i in range(Total_Indices):
            self.assertEqual(Hit_Counter[i], 1);



    def test_Col_Num_to_Multi_Index(self):
        # This function essentially checks that Col_Number_to_Multi_Index is
        # the inverse of Multi_Index_to_Col_Number. To test this, we check
        # that the composition of the two functions, in either order, gives
        # the identity map.

        Max_Sub_Indices      : int = random.randrange(1, 6);
        Num_Sub_Index_Values : int = random.randrange(2, 10);

        # First, we should generate a list of all multi-indices. We'll need this
        # for testing.
        Multi_Indices = [];
        Total_Indices = 0;
        for k in range(1, Max_Sub_Indices + 1):
            Num_k_Indices : int = Num_Multi_Indices(
                                    Num_Sub_Indices      = k,
                                    Num_Sub_Index_Values = Num_Sub_Index_Values);
            Total_Indices += Num_k_Indices;

            k_Indices = numpy.empty((Num_k_Indices, k), dtype = numpy.int64);
            Multi_Indices_Array(
                Multi_Indices        = k_Indices,
                Num_Sub_Indices      = k,
                Num_Sub_Index_Values = Num_Sub_Index_Values);

            Multi_Indices.append(k_Indices);

        # Initialize the Col_Num to Multi_Index map.
        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        Multi_Index_to_Col_Number = Multi_Index_to_Col_Number_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        # Test that Col_Number_to_Multi_Index maps column numbers to the correct
        # Indicies.
        for i in range(0, Total_Indices):
            Multi_Index_i = Col_Number_to_Multi_Index(i);
            Col_Num_i     = Multi_Index_to_Col_Number(Multi_Index_i);
            self.assertEqual(Col_Num_i, i);

            #print("%3d -> [" % i, end = '');
            #Num_Sub_Indices = Multi_Index_i.size;
            #for j in range(Num_Sub_Indices):
            #    print(" %2d " % Multi_Index_i[j], end = '');
            #print("] -> %3d" % Col_Num_i);
