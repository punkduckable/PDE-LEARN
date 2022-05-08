# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code, Classes directory to the python path.
Code_path       = os.path.join(parent_dir, "Code");
Classes_path    = os.path.join(parent_dir, "Classes");

# Append Code, Classes paths.
sys.path.append(Code_path);
sys.path.append(Classes_path);

# external libraries and stuff.
import numpy;
import torch;
import unittest;
import random;
import math;

# Code files.
from Evaluate_Derivatives   import Derivative_From_Derivative;
from Derivative             import Derivative;

# Other test file.
from Polynomials            import Polynomial_2D, Polynomial_3D;



class Test_Derivative_From_Derivative(unittest.TestCase):
    def test_Eval_Derivative_2D(self):
        # First, we need to set up a simple function with known derivatives so
        # that we can check it works properly. For this, I will use the
        # following function:
        #       (t, x) -> t^n + t^(n - 1) x + ... + t x^(n - 1) + x^n.
        n : int             = 3;
        P : Polynomial_2D   = Polynomial_2D(n);

        # Now, generate some coordinates.
        Num_Coords  : int           = 50;
        Coords      : torch.Tensor  = torch.empty((Num_Coords, 2));

        for i in range(Num_Coords):
            Coords[i, 0] = random.uniform(-1, 1);          # t
            Coords[i, 1] = random.uniform(-1, 1);          # x

        # Make sure the Coords require grad.
        Coords.requires_grad_(True);


        ########################################################################
        # Evaluate P, D_t P, D_t^2 P, D_x P, D_x^2 P, and D_x D_t P. We will
        # attempt to calculate the higher order derivatives in a number of ways
        # (directly from P, as well as from previous derivatives).

        t : torch.Tensor = Coords[:, 0];
        x : torch.Tensor = Coords[:, 1];

        # Compute P.
        P_Coords            = P(Coords).view(-1);

        # Compute D_t P.
        Dt_P_True : torch.Tensor = torch.zeros(Num_Coords);
        for i in range(0, n):
            Dt_P_True += (n - i)*torch.multiply(torch.pow(t, n - 1 - i), torch.pow(x, i));

        # Compute D_t^2 P.
        Dt2_P_True : torch.Tensor = torch.zeros(Num_Coords);
        for i in range(0, n - 1):
            Dt2_P_True += (n - i)*(n - 1 - i)*torch.multiply(torch.pow(t, n - 2 - i), torch.pow(x, i));

        # Compute D_x P.
        Dx_P_True : torch.Tensor = torch.zeros(Num_Coords);
        for i in range(1, n + 1):
            Dx_P_True += i*torch.multiply(torch.pow(t, n - i), torch.pow(x, i - 1));

        # Compute D_x^2 P.
        Dx2_P_True : torch.Tensor = torch.zeros(Num_Coords);
        for i in range(2, n + 1):
            Dx2_P_True += i*(i - 1)*torch.multiply(torch.pow(t, n - i), torch.pow(x, i - 2));

        # Compute D_x D_t P.
        DxDt_P_True : torch.Tensor = torch.zeros(Num_Coords);
        for i in range(1, n):
            DxDt_P_True += i*(n - i)*torch.multiply(torch.pow(t, n - 1 - i), torch.pow(x, i - 1));



        ########################################################################
        # Set up Derivative operators.

        I       : Derivative = Derivative(Encoding = numpy.array([0, 0]));
        Dt      : Derivative = Derivative(Encoding = numpy.array([1, 0]));
        Dt2     : Derivative = Derivative(Encoding = numpy.array([2, 0]));
        Dx      : Derivative = Derivative(Encoding = numpy.array([0, 1]));
        Dx2     : Derivative = Derivative(Encoding = numpy.array([0, 2]));
        Dx_Dt   : Derivative = Derivative(Encoding = numpy.array([1, 1]));

        epsilon : float = 1e-5;



        ########################################################################
        # Calculate Dx P, Dt P using Derivative_From_Derivative.

        Dx_P = Derivative_From_Derivative(Da = Dx, Db = I, Db_U = P_Coords, Coords = Coords);
        Dt_P = Derivative_From_Derivative(Da = Dt, Db = I, Db_U = P_Coords, Coords = Coords);

        for i in range(Num_Coords):
            # | Dx_P - Dx_P_True |, | Dt_P - Dt_P_True | (component-wise)
            Dx_Abs_Error = torch.abs(torch.subtract(Dx_P, Dx_P_True));
            Dt_Abs_Error = torch.abs(torch.subtract(Dt_P, Dt_P_True));

            # Check that there are no components of the above which are bigger
            # than epsilon.
            self.assertEqual(torch.sum(torch.greater_equal(Dx_Abs_Error, epsilon)), 0);
            self.assertEqual(torch.sum(torch.greater_equal(Dt_Abs_Error, epsilon)), 0);



        ########################################################################
        # Check that calculating Dx P from Dx P gives Dx P.

        Dx_P_From_Dx_P = Derivative_From_Derivative(Da = Dx, Db = Dx, Db_U = Dx_P, Coords = Coords);

        for i in range(Num_Coords):
            # | Dx_P_From_Dx_P - Dx_P_True | (component-wise)
            Dx_Abs_Error = torch.abs(torch.subtract(Dx_P_From_Dx_P, Dx_P_True));

            # Check that there are no components of the above which are bigger
            # than epsilon.
            self.assertEqual(torch.sum(torch.greater_equal(Dx_Abs_Error, epsilon)), 0);



        ########################################################################
        # Calculate Dx^2 P using Derivative_From_Derivative.

        # Calculate from P and D_x P.
        Dx2_P_From_P        = Derivative_From_Derivative(Da = Dx2, Db = I,  Db_U = P_Coords,    Coords = Coords);
        Dx2_P_From_Dx_P     = Derivative_From_Derivative(Da = Dx2, Db = Dx, Db_U = Dx_P,        Coords = Coords);

        # Check that they match Dx2_P_True.
        for i in range(Num_Coords):
            # | Dx2_P_From_P - Dx2_P_True |, | Dx2_P_From_Dx_P - Dx2_P_True |
            From_P_Abs_Error    = torch.abs(torch.subtract(Dx2_P_From_P,       Dx2_P_True));
            From_Dx_P_Abs_Error = torch.abs(torch.subtract(Dx2_P_From_Dx_P,    Dx2_P_True));

            # Check that there are no components of the above which are bigger
            # than epsilon.
            self.assertEqual(torch.sum(torch.greater_equal(From_P_Abs_Error,     epsilon)), 0);
            self.assertEqual(torch.sum(torch.greater_equal(From_Dx_P_Abs_Error,  epsilon)), 0);


        ########################################################################
        # Calculate Dt^2 P using Derivative_From_Derivative.

        # Calculate from P and D_t P.
        Dt2_P_From_P        = Derivative_From_Derivative(Da = Dt2, Db = I,  Db_U = P_Coords,    Coords = Coords);
        Dt2_P_From_Dt_P     = Derivative_From_Derivative(Da = Dt2, Db = Dt, Db_U = Dt_P,        Coords = Coords);

        # Check that they match Dx2_P_True.
        for i in range(Num_Coords):
            # | Dt2_P_From_P - Dt2_P_True |, | Dt2_P_From_Dx_P - Dt2_P_True |
            From_P_Abs_Error    = torch.abs(torch.subtract(Dt2_P_From_P,       Dt2_P_True));
            From_Dt_P_Abs_Error = torch.abs(torch.subtract(Dt2_P_From_Dt_P,    Dt2_P_True));

            # Check that there are no components of the above which are bigger
            # than epsilon.
            self.assertEqual(torch.sum(torch.greater_equal(From_P_Abs_Error,     epsilon)), 0);
            self.assertEqual(torch.sum(torch.greater_equal(From_Dt_P_Abs_Error,  epsilon)), 0);


        ########################################################################
        # Calculate Dt Dx P using Derivative_From_Derivative.

        # Calculate from P, D_t P, and D_x P.
        DxDt_P_From_P        = Derivative_From_Derivative(Da = Dx_Dt, Db = I,  Db_U = P_Coords,    Coords = Coords);
        DxDt_P_From_Dx_P     = Derivative_From_Derivative(Da = Dx_Dt, Db = Dx, Db_U = Dx_P,        Coords = Coords);
        DxDt_P_From_Dt_P     = Derivative_From_Derivative(Da = Dx_Dt, Db = Dt, Db_U = Dt_P,        Coords = Coords);

        # Check that they match DxDt_P_True.
        for i in range(Num_Coords):
            # | DxDt_P_From_P - DxDt_P_True |, | DxDt_P_From_Dx_P - DxDt_P_True |
            From_P_Abs_Error    = torch.abs(torch.subtract(DxDt_P_From_P,       DxDt_P_True));
            From_Dx_P_Abs_Error = torch.abs(torch.subtract(DxDt_P_From_Dx_P,    DxDt_P_True));
            From_Dt_P_Abs_Error = torch.abs(torch.subtract(DxDt_P_From_Dt_P,    DxDt_P_True));

            # Check that there are no components of the above which are bigger
            # than epsilon.
            self.assertEqual(torch.sum(torch.greater_equal(From_P_Abs_Error,     epsilon)), 0);
            self.assertEqual(torch.sum(torch.greater_equal(From_Dx_P_Abs_Error,  epsilon)), 0);
            self.assertEqual(torch.sum(torch.greater_equal(From_Dt_P_Abs_Error,  epsilon)), 0);


    """
    def test_Evalu_Deriv_3D(self):
        # First, we need to set up a simple function with known derivatives so that
        # we can check it works properly. For this, I will use the following
        # function:
        #       (t, x, y) -> t^n + x^n + x^(n-1)y + ... + xy^(n-1) + y^n
        n = 3;
        P = Polynomial_3D(n);

        # Now, generate some coordinates.
        Num_Coords : int    = 50;
        Coords              = torch.empty((Num_Coords, 3));

        for i in range(Num_Coords):
            Coords[i, 0] = random.uniform(-1, 1);          # t
            Coords[i, 1] = random.uniform(-1, 1);          # x
            Coords[i, 2] = random.uniform(-1, 1);          # y


        ########################################################################
        # Evaluate P and its derivatives at the Coordinates.

        Num_Sub_Indices : int   = Num_Sub_Index_Values_2D(3);
        Dxyn_P_true             = [None]*Num_Sub_Indices;

        # Aliases
        T = Coords[:, 0];
        X = Coords[:, 1];
        Y = Coords[:, 2];

        # P
        Col : int = xy_Derivatives_to_Index(0, 0);
        Dxyn_P_true[Col] = P(Coords).view(-1);

        # Dx_P, Dy_P
        Dxy1_P = torch.empty((Num_Coords, 2));

        Col_x : int = xy_Derivatives_to_Index(1, 0);
        Col_y : int = xy_Derivatives_to_Index(0, 1);

        for i in range(Num_Coords):
            Dxy1_P[i, 0] = 3*X[i]*X[i] + 2*X[i]*Y[i] + Y[i]*Y[i];
            Dxy1_P[i, 1] = X[i]*X[i] + 2*X[i]*Y[i] + 3*Y[i]*Y[i];

        Dxyn_P_true[Col_x] = Dxy1_P[:, 0].view(-1);
        Dxyn_P_true[Col_y] = Dxy1_P[:, 1].view(-1);


        # Dxx_P, Dxy_P, Dyy_P
        Dxy2_P = torch.empty((Num_Coords, 3));

        Col_xx : int = xy_Derivatives_to_Index(2, 0);
        Col_xy : int = xy_Derivatives_to_Index(1, 1);
        Col_yy : int = xy_Derivatives_to_Index(0, 2);

        for i in range(Num_Coords):
            Dxy2_P[i, 0] = 6*X[i] + 2*Y[i];
            Dxy2_P[i, 1] = 2*X[i] + 2*Y[i];
            Dxy2_P[i, 2] = 2*X[i] + 6*Y[i];

        Dxyn_P_true[Col_xx] = Dxy2_P[:, 0].view(-1);
        Dxyn_P_true[Col_xy] = Dxy2_P[:, 1].view(-1);
        Dxyn_P_true[Col_yy] = Dxy2_P[:, 2].view(-1);


        # Dxxx_P, Dxxy_P, Dxyy_P, Dyyy_P
        Dxy3_P = torch.empty((Num_Coords, 4));

        Col_xxx : int = xy_Derivatives_to_Index(3, 0);
        Col_xxy : int = xy_Derivatives_to_Index(2, 1);
        Col_xyy : int = xy_Derivatives_to_Index(1, 2);
        Col_yyy : int = xy_Derivatives_to_Index(0, 3);

        for i in range(Num_Coords):
            Dxy3_P[i, 0] = 6;
            Dxy3_P[i, 1] = 2;
            Dxy3_P[i, 2] = 2;
            Dxy3_P[i, 3] = 6;

        Dxyn_P_true[Col_xxx] = Dxy3_P[:, 0].view(-1);
        Dxyn_P_true[Col_xxy] = Dxy3_P[:, 1].view(-1);
        Dxyn_P_true[Col_xyy] = Dxy3_P[:, 2].view(-1);
        Dxyn_P_true[Col_yyy] = Dxy3_P[:, 3].view(-1);


        ########################################################################
        # Run Evaluate_Derivatives on P!
        (Dt_P, Dxyn_P) =  Evaluate_Derivatives(
                                U                                   = P,
                                Time_Derivative_Order               = 1,
                                Highest_Order_Spatial_Derivatives   = n,
                                Coords                              = Coords);


        ########################################################################
        # Check for match.
        epsilon : float = 1e-6;
        for k in range(Num_Sub_Indices):
            for i in range(Num_Coords):
                self.assertLess(abs(Dxyn_P_true[k][i].item() - Dxyn_P[k][i].item()), epsilon); """
