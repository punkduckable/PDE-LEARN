# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# external libraries and stuff.
import numpy;
import torch;
import unittest;
import random;
import math;

# Code files.
from Loss import Coll_Loss, L0_Approx_Loss, Lp_Loss;
from Points import Generate_Points;
from Evaluate_Derivatives import Evaluate_Derivatives;
from Mappings import xy_Derivatives_to_Index, Index_to_xy_Derivatives_Class, Index_to_x_Derivatives, \
                     Num_Multi_Indices, Multi_Indices_Array, \
                     Multi_Index_to_Col_Number_Class, Col_Number_to_Multi_Index_Class;

# Other test file.
from Polynomials import Polynomial_1d, Polynomial_2d;


class Loss_Test(unittest.TestCase):
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



    def test_Lp_Loss(self):
        ########################################################################
        # Test 1: Xi = 0.

        # Instantiate Xi.
        N : int   = random.randrange(5, 100);
        Xi        = torch.zeros(N, dtype = torch.float32);
        p : float = random.uniform(.01, .1);

        # In this case, we expect |Xi[0]|^p + ... |Xi[N-1]|^p = 0.
        Predict : float = 0;
        Actual  : float = Lp_Loss(Xi = Xi, p = p).item();

        # Check results
        epsilon : float = .00001;
        self.assertLess(abs(Predict - Actual), epsilon);


        ########################################################################
        # Test 2 : All componnts of Xi are the same.

        # Now replace Xi with a randomly selected value.
        x  = random.uniform(.01, .1);
        Xi = torch.full_like(Xi, x);

        # In this case, we expect the result to be N*(x^p).
        Predict = N*(x ** p);
        Actual  = Lp_Loss(Xi = Xi, p = p).item();

        # Check results
        self.assertLess(abs(Predict - Actual), epsilon);

        ########################################################################
        # Test 3 : a random number of components of Xi are the same, the rest
        # are zero.

        M : int = random.randrange(1, N - 1);
        Xi[M:] = 0;

        # In this case, we expect the result to be M*(x^p).
        Predict = M*(x **p );
        Actual = Lp_Loss(Xi = Xi, p = p).item();

        self.assertLess(abs(Predict - Actual), epsilon);
        #print("p = %f, x = %f, M = %d, Predict = %lf, actual = %f" % (p, x, M, Predict, Actual));


    def test_L0_Approx_Loss(self):
        ########################################################################
        # Test 1 : Xi = 0

        # Instantiate Xi.
        N : int = random.randrange(5, 100);
        Xi      = torch.empty(N, dtype = torch.float32);
        s       = random.uniform(.01, .1);

        # First, try with the zero vector.
        for i in range(N):
            Xi[i] = 0;

        # In this case, we expect the approximation to the L0 norm to give zero.
        Predict : float = 0;
        Actual  : float = L0_Approx_Loss(Xi = Xi, s = s).item();

        epsilon : float = .00001;
        self.assertLess(abs(Predict - Actual), epsilon);



        ########################################################################
        # Test 2 : All componnts of Xi are the same.

        # Now replace Xi with a randomly selected value.
        x  = random.uniform(.5, 1.5);
        Xi = torch.full_like(Xi, x);

        # In this case, we expect the result to be N(1 - exp(-x^2/s^2)).
        Predict = N*(1 - math.exp(-(x**2)/(s**2)));
        Actual  = L0_Approx_Loss(Xi = Xi, s = s).item();

        self.assertLess(abs(Predict - Actual), epsilon);



        ########################################################################
        # Test 3: Fill some, but not all, of Xi's components.

        Half_N = N // 2;
        Xi = torch.zeros_like(Xi);
        y = random.uniform(.01, .1);
        for i in range(Half_N):
            Xi[i] = y;

        # We now expect the result to be Half_N*(1 - exp(-y^2/(s**2))) (think about it).
        Predict = Half_N*(1 - math.exp(-(y**2)/(s**2)));
        Actual  = L0_Approx_Loss(Xi = Xi, s = s).item();

        self.assertLess(abs(Predict - Actual), epsilon);
        #print("s = %f, y = %f, N = %d, Predict = %lf, actual = %f" % (s, y, N, Predict, Actual));
