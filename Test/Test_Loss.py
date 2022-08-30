# Nonsense to add Code directory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_Path       = os.path.join(parent_dir, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Code, Classes paths.
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# external libraries and stuff.
import  numpy;
import  torch;
import  unittest;
import  random;
import  math;
from    typing  import List;

# Code files.
from    Derivative  import Derivative;
from    Term        import Term;
from    Loss        import Coll_Loss, L0_Approx_Loss, Lp_Loss;
from    Points      import Generate_Points;
from    Evaluate_Derivatives import Derivative_From_Derivative;

# Other test file.
from    Polynomials import Polynomial_2D, Polynomial_3D;



class Loss_Test(unittest.TestCase):
    def test_Coll_Loss_2D(self):
        # Set up U to be a 1d polynomial.
        n : int = 4;
        P       = Polynomial_2D(n);

        # Generate some collocation coordinates.
        Bounds = numpy.array(  [[0 , 1],
                                [-1, 1]], dtype = numpy.float32);
        Coords = Generate_Points(Bounds = Bounds, Num_Points = 1000);
        Coords.requires_grad_(True);

        # Evaluate predicted values. For this, we will use the following
        # library terms:
        #          (U)^4, (D_{x} U)^3, (D_{xx} U)^2, (D_{xxx} U), (U)(D_{x} U)^2

        # First, set up the Derivatives.
        I   : Derivative  = Derivative(Encoding = numpy.array([0, 0]));
        Dt  : Derivative  = Derivative(Encoding = numpy.array([1, 0]));
        Dx  : Derivative  = Derivative(Encoding = numpy.array([0, 1]));
        Dx2 : Derivative  = Derivative(Encoding = numpy.array([0, 2]));
        Dx3 : Derivative  = Derivative(Encoding = numpy.array([0, 3]));

        # Make the Derivatives list.
        Derivatives : List[Derivative] = [I, Dt, Dx, Dx2, Dx3];

        # Make the LHS Term.
        LHS_Term = Term(Derivatives = [Dt], Powers = [1]);

        # Make the RHS Terms.
        T_1 : Term = Term(Derivatives = [I],        Powers = [4]);
        T_2 : Term = Term(Derivatives = [Dx],       Powers = [3]);
        T_3 : Term = Term(Derivatives = [Dx2],      Powers = [2]);
        T_4 : Term = Term(Derivatives = [Dx3],      Powers = [1]);
        T_5 : Term = Term(Derivatives = [I, Dx],    Powers = [1, 2]);

        RHS_Terms : List[Term] = [T_1, T_2, T_3, T_4, T_5];

        # Now, evaluate each derivative of U at the Coords. (another test
        # verifies that this function works)
        P_Coords    = P(Coords).view(-1);
        Dt_P        = Derivative_From_Derivative(Da = Dt,  Db = I,   Db_U = P_Coords, Coords = Coords).view(-1);
        Dx_P        = Derivative_From_Derivative(Da = Dx,  Db = I,   Db_U = P_Coords, Coords = Coords).view(-1);
        Dx2_P       = Derivative_From_Derivative(Da = Dx2, Db = Dx,  Db_U = Dx_P,     Coords = Coords).view(-1);
        Dx3_P       = Derivative_From_Derivative(Da = Dx3, Db = Dx2, Db_U = Dx2_P,    Coords = Coords).view(-1);

        # For this test, Xi will be a random vector.
        Xi = torch.rand(5, dtype = torch.float32);

        # Calculate b(U) predicted.
        b_U_Pred    = Dt_P;

        # Calculate L(U)Xi predicted.
        L_U_Xi_Pred =  (Xi[0]*torch.pow(P_Coords, 4) +
                        Xi[1]*torch.pow(Dx_P,     3) +
                        Xi[2]*torch.pow(Dx2_P,    2) +
                        Xi[3]*Dx3_P +
                        Xi[4]*torch.multiply(P_Coords, torch.pow(Dx_P, 2)));

        # Calculated predicted loss (mean square residual).
        Loss_Pred = torch.mean(torch.pow(torch.subtract(b_U_Pred, L_U_Xi_Pred), 2));



        ########################################################################
        # Now let's see what Coll_Loss actually gives.

        Loss_Actual = Coll_Loss(    U               = P,
                                    Xi              = Xi,
                                    Coll_Points     = Coords,
                                    Derivatives     = Derivatives,
                                    LHS_Term        = LHS_Term,
                                    RHS_Terms       = RHS_Terms)[0];
        # Check that it worked!
        self.assertEqual(Loss_Actual.item(), Loss_Pred.item());

    """
    def test_Coll_Loss_3D(self):
        # Set up U to be a 2d polynomial.
        n : int = 3;
        U_Poly = Polynomial_3D(n);

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
        Highest_Order_Spatial_Derivatives : int = 2;
        Max_Sub_Indices : int = 1;
        (Dt_U, Dxy_U)         = Evaluate_Derivatives(
                                    U                                   = U_Poly,
                                    Time_Derivative_Order               = 1,
                                    Highest_Order_Spatial_Derivatives   = Highest_Order_Spatial_Derivatives,
                                    Coords                              = Coll_Points);

        # Evaluate Library_Xi product!
        #                     U          D_{x}U     D_{y}U
        Library_Xi_Product = (Dxy_U[0] + Dxy_U[1] + Dxy_U[2] +
                              Dxy_U[3] + Dxy_U[4] + Dxy_U[5] + torch.ones_like(Dt_U));
        #                     D_{xx}U    D_{xy}U    D_{yy}U    1

        # Now evaluate the mean square difference between Dt_U and the
        # Library_Xi_Product.
        Square_Error_Predict = torch.pow(torch.sub(Dt_U, Library_Xi_Product), 2);
        Coll_Loss_Predict    = Square_Error_Predict.mean();


        ########################################################################
        # Now let's see what Coll_Loss actually gives.

        # Initialize Xi, Col_Number_to_Multi_Index and Index_to_xy_Derivatives.
        Xi = torch.ones(7, dtype = torch.float32);

        Index_to_xy_Derivatives = Index_to_xy_Derivatives_Class(
                                        Highest_Order_Derivatives = Highest_Order_Spatial_Derivatives);

        Num_Sub_Index_Values = Index_to_xy_Derivatives.Num_Index_Values;
        Col_Number_to_Multi_Index = Col_Number_to_Multi_Index_Class(
                                        Max_Sub_Indices      = Max_Sub_Indices,
                                        Num_Sub_Index_Values = Num_Sub_Index_Values);

        Coll_Loss_Actual = Coll_Loss(   U                                   = U_Poly,
                                        Xi                                  = Xi,
                                        Coll_Points                         = Coll_Points,
                                        Time_Derivative_Order               = 1,
                                        Highest_Order_Spatial_Derivatives   = Highest_Order_Spatial_Derivatives,
                                        Index_to_Derivatives                = Index_to_xy_Derivatives,
                                        Col_Number_to_Multi_Index           = Col_Number_to_Multi_Index);
        # Check that it worked!
        self.assertEqual(Coll_Loss_Actual, Coll_Loss_Predict);"""



    def test_Lp_Loss(self):
        ########################################################################
        # Test 1: Xi = 0.

        # Instantiate Xi.
        N : int   = random.randrange(5, 100);
        Xi        = torch.zeros(N, dtype = torch.float32);
        p : float = random.uniform(.01, .1);

        # In this case, we expect the Lp loss to be 0.
        Predict : float = 0;
        Actual  : float = Lp_Loss(Xi = Xi, p = p).item();

        # Check results
        epsilon : float = .00001;
        self.assertLess(abs(Predict - Actual), epsilon);


        ########################################################################
        # Test 2 : All components of Xi are the same.

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
        # Test 2 : All components of Xi are the same.

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
