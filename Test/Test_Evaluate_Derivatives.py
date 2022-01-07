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
from Evaluate_Derivatives import Evaluate_Derivatives;
from Mappings import    x_Derivatives_to_Index, xy_Derivatives_to_Index, \
                        Num_Sub_Index_Values_1D, Num_Sub_Index_Values_2D;

# Other test file.
from Polynomials import Polynomial_1d, Polynomial_2d;



class Test_Evaluate_Derivatives(unittest.TestCase):
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

        Num_Sub_Indices : int   = Num_Sub_Index_Values_1D(3);
        Dxyn_P_true             = [None]*Num_Sub_Indices;

        # Aliases
        T = Coords[:, 0];
        X = Coords[:, 1];

        # P
        Col : int           = x_Derivatives_to_Index(0);
        Dxyn_P_true[Col]    = P(Coords).view(-1);

        # Dx_P
        Col : int   = x_Derivatives_to_Index(1);
        Dxy1_P      = torch.empty(Num_Coords);
        for i in range(Num_Coords):
            Dxy1_P[i] = 3*X[i]*X[i] + 2*X[i] + 1;
        Dxyn_P_true[Col] = Dxy1_P;

        # Dxx_P
        Col : int   = x_Derivatives_to_Index(2);
        Dxy2_P      = torch.empty(Num_Coords);
        for i in range(Num_Coords):
            Dxy2_P[i]       = 6*X[i] + 2;
        Dxyn_P_true[Col] = Dxy2_P;

        # Dxxx_P
        Col : int   = x_Derivatives_to_Index(3);
        Dxy3_P      = torch.empty(Num_Coords);
        for i in range(Num_Coords):
            Dxy3_P[i] = 6;
        Dxyn_P_true[Col] = Dxy3_P;


        ########################################################################
        # Run Evaluate_Derivatives on P!
        (Dt_P, Dxyn_P) =  Evaluate_Derivatives(
                                U = P,
                                Highest_Order_Derivatives = n,
                                Coords = Coords);


        ########################################################################
        # Check for match.
        epsilon : float = 1e-6;
        for k in range(Num_Sub_Indices):
            for i in range(Num_Coords):
                self.assertLess(abs(Dxyn_P_true[k][i].item() - Dxyn_P[k][i].item()), epsilon);



    def test_Evalu_Deriv_2D(self):
        # First, we need to set up a simple function with known derivatives so that
        # we can check it works properly. For this, I will use the following
        # function:
        #       (t, x, y) -> t^n + x^n + x^(n-1)y + ... + xy^(n-1) + y^n
        n = 3;
        P = Polynomial_2d(n);

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
                                U = P,
                                Highest_Order_Derivatives = n,
                                Coords = Coords);


        ########################################################################
        # Check for match.
        epsilon : float = 1e-6;
        for k in range(Num_Sub_Indices):
            for i in range(Num_Coords):
                self.assertLess(abs(Dxyn_P_true[k][i].item() - Dxyn_P[k][i].item()), epsilon);
