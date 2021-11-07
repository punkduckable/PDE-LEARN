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
from Mappings import xy_Derivatives_to_Index, Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, \
                     Num_Multi_Indices, Multi_Indices_Array, \
                     Multi_Index_to_Col_Number_Class, Col_Number_to_Multi_Index_Class;
from Points import Generate_Points;
from Loss import Coll_Loss, Lp_Loss;


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



class Coll_Loss_Test(unittest.TestCase):
    def test_Coll_Loss_1D(self):
        # Set up U to be a 1d polynomial.
        n : int = 4;
        U_Poly = Polynomial_1d(n);

        # Generate some collocation coordinates.
        Bounds = numpy.array(  [[0 , 1],
                                [-1, 1]], dtype = numpy.float32);
        Coll_Points = Generate_Points(Bounds = Bounds, Num_Points = 1000);

        # Evaluate predicted values. For this, we will use derivatives of up to
        # order 4. We will also only allow linear terms. In this case, there are
        # 6 library terms:
        #          U, D_{x}U, D_{xx}U, D_{xxx}U, D_{xxxx}U, 1
        # We evaluate each one using Evaluate_Derivatives (another test
        # verifies that this function works). We Also assume Xi is a vector of
        # ones. As such, we should get the following Library-Xi product.
        #       U + D_{x}U + D_{xx}U + D_{xxx}U + D_{xxxx}U + 1
        # And thus, we expect the Coll_Loss to be the mean of the square of
        # the difference between this value and D_{t}U.
        Highest_Order_Derivatives : int = 4;
        Max_Sub_Indices : int = 1;
        (Dt_U, Dxy_U)         = Evaluate_Derivatives(
                                    U = U_Poly,
                                    Highest_Order_Derivatives = Highest_Order_Derivatives,
                                    Coords = Coll_Points);

        # Evaluate Library_Xi product!
        #                     U                D_{x}U           D_{xx}U
        Library_Xi_Product = (Dxy_U[0][:, 0] + Dxy_U[1][:, 0] + Dxy_U[2][:, 0] +
                              Dxy_U[3][:, 0] + Dxy_U[4][:, 0] + torch.ones_like(Dt_U));
        #                     D_{xxx}U          D_{xxxx}U       1

        # Now evaluate the mean square difference between Dt_U and the
        # Library_Xi_Product.
        Square_Error_Predict = torch.pow(torch.sub(Dt_U, Library_Xi_Product), 2);
        Coll_Loss_Predict    = Square_Error_Predict.mean();


        ########################################################################
        # Now let's see what Coll_Loss actually gives.

        # Initialize Xi, Col_Number_to_Multi_Index and Index_to_xy_Derivatives.
        Xi = torch.ones(6, dtype = torch.float32);

        Num_Sub_Index_Values = 5; # U, D_{x}U, D_{xx}U, D_{xxx}U, D_{xxxx}U
        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        Coll_Loss_Actual = Coll_Loss(   U           = U_Poly,
                                        Xi          = Xi,
                                        Coll_Points = Coll_Points,
                                        Highest_Order_Derivatives = Highest_Order_Derivatives,
                                        Index_to_Derivatives      = Index_to_x_Derivatives,
                                        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index);
        # Check that it worked!
        self.assertEqual(Coll_Loss_Actual, Coll_Loss_Predict);

    def test_Coll_Loss_2D(self):
        # Set up U to be a 2d polynomial.
        n : int = 3;
        U_Poly = Polynomial_2d(n);

        # Generate some collocation coordinates.
        Bounds = numpy.array(  [[0 , 1],
                                [-1, 1],
                                [-1, 1]], dtype = numpy.float32);
        Coll_Points = Generate_Points(Bounds = Bounds, Num_Points = 1000);

        # Evaluate predicted values. For this, we will use derivative of up to
        # order 2. We will also only allow linear terms. In this case,
        # there are 7 library terms:
        #          U, D_{x}U, D_{y}U, D_{xx}U, D_{xy}U, D_{yy}U, 1
        # We evaluate each one using Evaluate_Derivatives (another test
        # verifies that this function works). We Also assume Xi is a vector of
        # ones. As such, we should get the following Library-Xi product.
        #       U + D_{x}U + D_{y}U + D_{xx}U + D_{xy}U + D_{yy}U + 1
        # And thus, we expect the Coll_Loss to be the mean of the square of
        # the difference between this value and D_{t}U.
        Highest_Order_Derivatives : int = 2;
        Max_Sub_Indices : int = 1;
        (Dt_U, Dxy_U)         = Evaluate_Derivatives(
                                    U = U_Poly,
                                    Highest_Order_Derivatives = Highest_Order_Derivatives,
                                    Coords = Coll_Points);

        # Evaluate Library_Xi product!
        #                     U                D_{x}U           D_{xx}}U
        Library_Xi_Product = (Dxy_U[0][:, 0] + Dxy_U[1][:, 0] + Dxy_U[1][:, 1] +
                              Dxy_U[2][:, 0] + Dxy_U[2][:, 1] + Dxy_U[2][:, 2] + torch.ones_like(Dt_U));
        #                     D_{xx}U          D_{xy}U          D_{yy}U         1

        # Now evaluate the mean square difference between Dt_U and the
        # Library_Xi_Product.
        Square_Error_Predict = torch.pow(torch.sub(Dt_U, Library_Xi_Product), 2);
        Coll_Loss_Predict    = Square_Error_Predict.mean();


        ########################################################################
        # Now let's see what Coll_Loss actually gives.

        # Initialize Xi, Col_Number_to_Multi_Index and Index_to_xy_Derivatives.
        Xi = torch.ones(7, dtype = torch.float32);

        Index_to_xy_Derivatives = Index_to_xy_Derivatives_Class(
                                        Highest_Order_Derivatives = Highest_Order_Derivatives);

        Num_Sub_Index_Values = Index_to_xy_Derivatives.Num_Index_Values;
        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        Coll_Loss_Actual = Coll_Loss(   U           = U_Poly,
                                        Xi          = Xi,
                                        Coll_Points = Coll_Points,
                                        Highest_Order_Derivatives = Highest_Order_Derivatives,
                                        Index_to_Derivatives      = Index_to_xy_Derivatives,
                                        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index);
        # Check that it worked!
        self.assertEqual(Coll_Loss_Actual, Coll_Loss_Predict);



if __name__ == "__main__":
    unittest.main();
