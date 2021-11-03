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

from Evaluate_Derivatives import Evaluate_Derivatives;
from Mappings import xy_Derivatives_To_Index, Index_to_xy_Derivatives, \
                     Num_Multi_Indices, Multi_Indices_Array, Multi_Index_To_Col_Number, \
                     Col_Number_To_Multi_Index;



class Polynomial_1d:
    def __init__(self, n : int):
        self.n = n;
        self.Input_Dim = 2;

    def __call__(self, Coords : torch.tensor):
        # Alias the three columns of Coords.
        t = Coords[:, 0];
        x = Coords[:, 1];

        Num_Coords = x.shape[0];
        P_XY = torch.zeros((Num_Coords, 1));

        P_XY += t.pow(self.n).view(-1, 1);
        for i in range(0, self.n + 1):
            P_XY += x.pow(self.n - i).view(-1, 1);

        return P_XY;



class Polynomial_2d:
    def __init__(self, n : int):
        self.n = n;
        self.Input_Dim = 3;

    def __call__(self, Coords : torch.tensor):
        # Alias the three columns of Coords.
        t = Coords[:, 0];
        x = Coords[:, 1];
        y = Coords[:, 2];

        Num_Coords = x.shape[0];
        P_XY = torch.zeros((Num_Coords, 1));

        P_XY += t.pow(self.n).view(-1, 1);
        for i in range(0, self.n + 1):
            P_XY += ((x.pow(self.n - i))*(y.pow(i))).view(-1, 1);

        return P_XY;



class Test_Eval_Deriv(unittest.TestCase):
    def test_Evalu_Deriv_1D(self):
        # First, we need to set up a simple function with known derivatives so that
        # we can check it works properly. For this, I will use the following
        # function:
        #       (t, x) -> t^n + x^n + x^(n - 1) + ... + x + 1
        n = 3;
        P = Polynomial_1d(n);

        # Now, generate some coordinates.
        Num_Coords : int = 50;
        Coords = torch.empty((Num_Coords, 2));

        for i in range(Num_Coords):
            Coords[i, 0] = random.uniform(-1, 1);          # t
            Coords[i, 1] = random.uniform(-1, 1);          # x


        ########################################################################
        # Evaluate P and its derivatives at the Coordinates.

        Dxyn_P_true = [];

        # Aliases
        T = Coords[:, 0];
        X = Coords[:, 1];

        # P
        Dxyn_P_true.append(P(Coords).view(-1, 1));

        # Dx_P
        Dxy1_P = torch.empty((Num_Coords, 1));
        for i in range(Num_Coords):
            Dxy1_P[i, 0] = 3*X[i]*X[i] + 2*X[i] + 1;
        Dxyn_P_true.append(Dxy1_P);

        # Dxx_P
        Dxy2_P = torch.empty((Num_Coords, 1));
        for i in range(Num_Coords):
            Dxy2_P[i, 0] = 6*X[i] + 2;
        Dxyn_P_true.append(Dxy2_P);

        # Dxxx_P
        Dxy3_P = torch.empty((Num_Coords, 1));
        for i in range(Num_Coords):
            Dxy3_P[i, 0] = 6;
        Dxyn_P_true.append(Dxy3_P);


        ########################################################################
        # Run Evaluate_Derivatives on P!
        (Dt_P, Dxyn_P) =  Evaluate_Derivatives(
                                U = P,
                                Highest_Order_Derivatives = n,
                                Coords = Coords);


        ########################################################################
        # Check for match.
        epsilon : float = 1e-6;
        for k in range(n):
            for i in range(Num_Coords):
                self.assertLess(abs(Dxyn_P_true[k][i, 0].item() - Dxyn_P[k][i, 0].item()), epsilon);



    def test_Evalu_Deriv_2D(self):
        # First, we need to set up a simple function with known derivatives so that
        # we can check it works properly. For this, I will use the following
        # function:
        #       (t, x, y) -> t^n + x^n + x^(n-1)y + ... + xy^(n-1) + y^n
        n = 3;
        P = Polynomial_2d(n);

        # Now, generate some coordinates.
        Num_Coords : int = 50;
        Coords = torch.empty((Num_Coords, 3));

        for i in range(Num_Coords):
            Coords[i, 0] = random.uniform(-1, 1);          # t
            Coords[i, 1] = random.uniform(-1, 1);          # x
            Coords[i, 2] = random.uniform(-1, 1);          # y


        ########################################################################
        # Evaluate P and its derivatives at the Coordinates.

        Dxyn_P_true = [];

        # Aliases
        T = Coords[:, 0];
        X = Coords[:, 1];
        Y = Coords[:, 2];

        # P
        Dxyn_P_true.append(P(Coords).view(-1, 1));

        # Dx_P, Dy_P
        Dxy1_P = torch.empty((Num_Coords, 2));
        for i in range(Num_Coords):
            Dxy1_P[i, 0] = 3*X[i]*X[i] + 2*X[i]*Y[i] + Y[i]*Y[i];
            Dxy1_P[i, 1] = X[i]*X[i] + 2*X[i]*Y[i] + 3*Y[i]*Y[i];
        Dxyn_P_true.append(Dxy1_P);

        # Dxx_P, Dxy_P, Dyy_P
        Dxy2_P = torch.empty((Num_Coords, 3));
        for i in range(Num_Coords):
            Dxy2_P[i, 0] = 6*X[i] + 2*Y[i];
            Dxy2_P[i, 1] = 2*X[i] + 2*Y[i];
            Dxy2_P[i, 2] = 2*X[i] + 6*Y[i];
        Dxyn_P_true.append(Dxy2_P);

        # Dxxx_P, Dxxy_P, Dxyy_P, Dyyy_P
        Dxy3_P = torch.empty((Num_Coords, 4));
        for i in range(Num_Coords):
            Dxy3_P[i, 0] = 6;
            Dxy3_P[i, 1] = 2;
            Dxy3_P[i, 2] = 2;
            Dxy3_P[i, 3] = 6;
        Dxyn_P_true.append(Dxy3_P);


        ########################################################################
        # Run Evaluate_Derivatives on P!
        (Dt_P, Dxyn_P) =  Evaluate_Derivatives(
                                U = P,
                                Highest_Order_Derivatives = n,
                                Coords = Coords);


        ########################################################################
        # Check for match.
        epsilon : float = 1e-6;
        for k in range(n):
            for j in range(k + 1):
                for i in range(Num_Coords):
                    self.assertLess(abs(Dxyn_P_true[k][i, j].item() - Dxyn_P[k][i,j].item()), epsilon);



class xy_Derivatives_And_Index(unittest.TestCase):
    def test_xy_Derivatives_to_Index(self):
        # Test that the function gives the expected output for derivatives less
        # than a specified order, M. Note that we do not care about the order in
        # which xy_Derivatives_To_Index maps elements of N^2 to N, so long as
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
                Index : int = xy_Derivatives_To_Index(i, j);

                # check that the Index is in {0, 1,... (M + 1)(M + 2)/2 - 1}.
                self.assertGreaterEqual(Index, 0);
                self.assertLess(Index, Num_Index_Values);

                Hit_Counter[Index] += 1;

                #print("(%d, %d) -> %d" % (i, j, Index));

        # now check that each value in {0, 1,... (M + 1)(M + 2)/2} got hit once.
        for i in range(Num_Index_Values):
            self.assertEqual(Hit_Counter[i], 1);



    def test_Index_To_xy_Derivatives_Array(self):
        # Generate a partial inverse, then test that this inverse actually
        # works.

        # First, pick a random index (with a reasonable value).
        Max_Derivatives : int = random.randrange(1, 10);

        # Now generate the inverse list.
        Index_To_Deriv = Index_to_xy_Derivatives(Max_Derivatives);

        # Determine maximum index value that we inverted.
        N = Index_To_Deriv.Num_Index_Values;

        # Check that Index_to_xy_Derivatives composed with
        # xy_Derivatives_To_Index is the identity.
        for k in range(N):
            xy_Deriv = Index_To_Deriv(k).reshape(-1);
            i : int = xy_Deriv[0];
            j : int = xy_Deriv[1];
            Index_Num : int = xy_Derivatives_To_Index(i, j);
            self.assertEqual(k, Index_Num);

            #print("%d -> (%d, %d) -> %d" % (k, i, j, Index_Num));

        # check that xy_Derivatives_To_Index composed with
        # Index_to_xy_Derivatives is the identity.
        for k in range(0, Max_Derivatives):
            for j in range(0, k + 1):
                i : int = k - j;        # Number of x derivatives.

                Index : int = xy_Derivatives_To_Index(i, j);
                xy_Der = Index_To_Deriv(Index).reshape(-1);

                self.assertEqual(xy_Der[0].item(), i);
                self.assertEqual(xy_Der[1].item(), j);

                #print("(%d, %d) -> %d -> (%d, %d)" % (i, j, Index, xy_Der[0].item(), xy_Der[1].item()));



class Multi_Index_And_Col_Number(unittest.TestCase):
    def test_Multi_Index_To_Col_Number(self):
        # First, determine how many sub-indices we want to use, and how many
        # index values we want per sub-index.
        Max_Sub_Indices      : int = random.randrange(1, 6);
        Num_Sub_Index_Values : int = random.randrange(2, 15);

        MI_To_Col = Multi_Index_To_Col_Number(
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
                Col_Num : int = MI_To_Col(Multi_Indices[k][j, :]);
                self.assertLess(Col_Num, Total_Indices);
                Hit_Counter[Col_Num] += 1;

                #Multi_Index = Multi_Indices[k][j, :];
                #print("[ ", end = '');
                #for i in range(0, k + 1):
                #    print("%2d " % Multi_Index[i], end = '');
                #print("] -> %d" % MI_To_Col(Multi_Index));

        # Check that each element of the range is hit exactly once.
        for i in range(Total_Indices):
            self.assertEqual(Hit_Counter[i], 1);



    def test_Col_Num_To_Multi_Index(self):
        # This function essentially checks that Col_Number_To_Multi_Index is
        # the inverse of Multi_Index_To_Col_Number. To test this, we check
        # that the composition of the two functions, in either order, gives
        # the identity map.

        Max_Sub_Indices : int      = random.randrange(1, 6);
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
        Col_To_MI = Col_Number_To_Multi_Index(
                        Max_Sub_Indices      = Max_Sub_Indices,
                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        MI_To_Col = Multi_Index_To_Col_Number(
                        Max_Sub_Indices      = Max_Sub_Indices,
                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        # Test that Col_Number_To_Multi_Index maps column numbers to the correct
        # Indicies.
        for i in range(0, Total_Indices):
            Multi_Index_i = Col_To_MI(i);
            Col_Num_i     = MI_To_Col(Multi_Index_i);
            self.assertEqual(Col_Num_i, i);

            #print("%3d -> [" % i, end = '');
            #Num_Sub_Indices = Multi_Index_i.size;
            #for j in range(Num_Sub_Indices):
            #    print(" %2d " % Multi_Index_i[j], end = '');
            #print("] -> %3d" % Col_Num_i);



if __name__ == "__main__":
    unittest.main();
