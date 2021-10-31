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

from Evaluate_Derivatives import Evaluate_Derivatives;



class Polynomial:
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



def Test_Eval_Deriv_2d():
    # First, we need to set up a simple function with known derivatives so that
    # we can check it works properly. For this, I will use the following
    # function:
    #       (t, x, y) -> t^n + x^n + x^(n-1)y + ... + xy^(n-1) + y^n
    n = 3;
    P = Polynomial(3);

    # Now, generate some coordinates to evaluate P.
    Coord_Side_Length : int = 3;
    Coords = torch.empty((Coord_Side_Length**3, 3));
    Position = 0;
    for i in range(Coord_Side_Length):
        for j in range(Coord_Side_Length):
            for k in range(Coord_Side_Length):
                Coords[Position, 0] = i;
                Coords[Position, 1] = j;
                Coords[Position, 2] = k;

                Position += 1;

    # Run Evaluate_Derivatives on P!
    (Dt_P, Dxyn_P) =  Evaluate_Derivatives(
                            U = P,
                            Highest_Order_Derivatives = n,
                            Coords = Coords);

    # Display results for manual inspection.
    Position = 0;
    for i in range(Coord_Side_Length):
        for j in range(Coord_Side_Length):
            for k in range(Coord_Side_Length):
                print("      P(%3.1f, %3.1f, %3.1f) = %.2f" % (Coords[Position, 0], Coords[Position, 1], Coords[Position, 2], Dxyn_P[0][Position, 0].item()));
                print("D_{t} P(%3.1f, %3.1f, %3.1f) = %.2f" % (Coords[Position, 0], Coords[Position, 1], Coords[Position, 2], Dt_P[Position].item()));
                Position += 1;

if __name__ == "__main__":
    Test_Eval_Deriv_2d();
