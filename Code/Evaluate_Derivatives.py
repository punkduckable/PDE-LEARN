# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Classes_Path);

from    typing      import Tuple, List;
import  torch;
import  numpy;

from    Derivative  import Derivative;



def Derivative_From_Derivative(
        Da          : Derivative,
        Db          : Derivative,
        Db_U        : torch.Tensor,
        Coords      : torch.Tensor) -> torch.Tensor:
    """ 
    This function applies the derivative operator Da to U. It does so,
    however, by calculating Da U from Db U, where Db is a partial derivative
    operator that is "child derivative" of Da. This means that Da's Encoding
    vector is >= (element-wise ) and Db's. If this is the case, then we can
    compute Da U from Db U, which is precisely what this function does. To see
    how, suppose that Da.Encoding = [p_1, ... , p_n], and Db.Encoding = [q_1,
    ... , q_n] where each p_k >= q_k. Further, let x(k) denote the kth variable.
    In this case,
            Da U = D_{x(1)}^{r(1)} ... D_{x(n)}^{r(n)} Db U
    This allows us to calculate Da U without having to re-do all the
    computations needed to compute Db U.

    Note: This function assumes that we computed Db U at Coords. This function
    also assumes the graph from Coords to Db_U exists (we need this to compute
    further derivatives). Finally, this function assumes Coords has
    requires grad set to true (and, in particular, had this set to true you
    computed Db_U)

    ----------------------------------------------------------------------------
    Arguments:

    Da : A derivative operator. This function computes Da U at Coords.

    Db : A derivative operator. Db must be a child of Da using the definition
    above.

    Db_U : This should be a 1D tensor whose ith entry holds the value of Db_U
    evaluated at the ith coordinate. We assume that the graph to get from Coords
    to Db U exists.

    Coords : A 2D tensor whose ith entry holds the ith coordinate at which we
    want to evaluate Da U. The ith entry of Db_U should hold the value of Db U
    at the ith coordinate. This function computes Da U at the coordinates.

    ----------------------------------------------------------------------------
    Returns:

    This returns a 1D Tensor. If Coords has B rows, then the returned tensor has
    B elements. The ith component of the returned Tensor holds Da U evaluated
    at the ith coordinate (ith row of Coords). 
    """

    # First, make sure that Db is a child of Da.
    assert(Db.Is_Child_Of(Da));

    # Now, let's get to work. The plan is the following: Since Db is a child of
    # Da, there is some derivative operator, Dc, such that
    #       Da = Dc Db
    # In particular, if Da.Encoding = [p_1, ... , p_l] and Db.Encoding = [q_1,
    # ... , q_l], then Dc.Encoding = [p_1 - q_1, ... , p_l - q_l]. Thus, to 
    # compute Da U from Db U, we first compute
    #       D_t^{p_1 - q_1} Da U.
    # From this, we calculate
    #        D_x^{p_2 - q_2} D_t^{p_1 - q_1} Db U.
    # And so on.

    # To make things easier, if Da.Encoding is shorter than Db.Encoding, we pad
    # the end of Da.Encoding with zeros until the two have the same length. We 
    # do the same thing to Db.Encoding if it is shorter than Da.Encoding.
    n : int = len(Da.Encoding);
    m : int = len(Db.Encoding);
    l : int = max(n, m);

    Da_Encoding_Padded : numpy.ndarray = numpy.zeros(l, dtype = numpy.int32);
    Db_Encoding_Padded : numpy.ndarray = numpy.zeros(l, dtype = numpy.int32);

    for k in range(n):
        Da_Encoding_Padded[k] = Da.Encoding[k];
    for k in range(m):
        Db_Encoding_Padded[k] = Db.Encoding[k];


    ############################################################################
    # t derivatives.

    # Initialize Dt_Db_U. If there are no t derivatives, then Dt_Db_U = Db_U.
    Dt_Db_U : torch.Tensor = Db_U;

    Dt_Order : int = Da_Encoding_Padded[0] - Db_Encoding_Padded[0];
    if(Dt_Order > 0):
        # Suppose Dt_Order = m. Compute D_t^k Db_U from D_t^{k - 1} Db_U for
        # each k in {1, 2, ... , m}.
        for k in range(1, Dt_Order + 1):
            # Compute the gradient.
            Grad_Dt_Db_U : torch.Tensor = torch.autograd.grad(
                                    outputs         = Dt_Db_U,
                                    inputs          = Coords,
                                    grad_outputs    = torch.ones_like(Dt_Db_U),
                                    retain_graph    = True,
                                    create_graph    = True)[0];

            # Update Dt_w (this replaces D_t^{k - 1} w with D_t^k w)
            Dt_Db_U = Grad_Dt_Db_U[:, 0].view(-1);



    ############################################################################
    # x derivatives.

    # Initialize Dx_Dt_Db_U. If there are no x derivatives, then Dx_Dt_Db_U
    # = Dt_Db_U.
    Dx_Dt_Db_U : torch.Tensor = Dt_Db_U;

    Dx_Order : int = Da_Encoding_Padded[1] - Db_Encoding_Padded[1];
    if(Dx_Order > 0):
        # Suppose Dx_Order = m. We compute D_x^k Dt_Db_U from
        # D_t^{k - 1} Dt_Db_U for each k in {1, 2, ... , m}.
        for k in range(1, Dx_Order + 1):
            # Compute the gradient.
            Grad_Dx_Dt_Db_U : torch.Tensor = torch.autograd.grad(
                                    outputs         = Dx_Dt_Db_U,
                                    inputs          = Coords,
                                    grad_outputs    = torch.ones_like(Dx_Dt_Db_U),
                                    retain_graph    = True,
                                    create_graph    = True)[0];

            # Update Dx_Dt_Db_U (this replaces D_x^{k - 1} Dt_Db_U with D_x^k Dt_Db_U)
            Dx_Dt_Db_U = Grad_Dx_Dt_Db_U[:, 1].view(-1);



    ############################################################################
    # y derivatives.

    # First, check if there are any y derivatives (if l >= 3). If not, then
    # we're done.
    if(l < 3):
        return Dx_Dt_Db_U;
    
    # Assuming we need y derivatives, initialize Dy_Dx_Dt_Db_U. If there are no
    # y derivatives, then Dy_Dx_Dt_Db_U = Dx_Dt_Db_U.
    Dy_Dx_Dt_Db_U : torch.Tensor = Dx_Dt_Db_U;

    Dy_Order : int = Da_Encoding_Padded[2] - Db_Encoding_Padded[2];
    if(Dy_Order > 0):
        # Suppose Dy_Order = m. We compute D_y^k Dx_Dt_Db_U from
        # D_y^{k - 1} Dx_Dt_Db_U for each k in {1, 2, ... , m}.
        for k in range(1, Dy_Order + 1):
            # Compute the gradient.
            Grad_Dy_Dx_Dt_Db_U : torch.Tensor = torch.autograd.grad(
                                    outputs         = Dy_Dx_Dt_Db_U,
                                    inputs          = Coords,
                                    grad_outputs    = torch.ones_like(Dy_Dx_Dt_Db_U),
                                    retain_graph    = True,
                                    create_graph    = True)[0];

            # Update Dy_Dx_Dt_Db_U (this replaces D_y^{k - 1} Dx_Dt_Db_U with
            # D_y^k Dx_Dt_Db_U)
            Dy_Dx_Dt_Db_U = Grad_Dy_Dx_Dt_Db_U[:, 2].view(-1);



    ############################################################################
    # z derivatives.

    # First, check if there are any z derivatives (if Derivative.Encoding has a
    # 4th element). If not, then we're done.
    if(l < 4):
        return Dy_Dx_Dt_Db_U;

    # Assuming we need z derivatives, initialize Dz_Dy_Dx_Dt_Db_U. If there are no
    # z derivatives, then Dz_Dy_Dx_Dt_Db_U = Dy_Dx_Dt_Db_U.
    Dz_Dy_Dx_Dt_Db_U : torch.Tensor = Dy_Dx_Dt_Db_U;

    Dz_Order : int = Da_Encoding_Padded[3] - Db_Encoding_Padded[3];
    if(Dz_Order > 0):
        # Suppose Dz_Order = m. We compute D_z^k Dy_Dx_Dt_Db_U from
        # D_z^{k - 1} Dy_Dx_Dt_Db_U for each k in {1, 2, ... , m}.
        for k in range(1, Dz_Order + 1):
            # Compute the gradient.
            Grad_Dz_Dy_Dx_Dt_Db_U : torch.Tensor = torch.autograd.grad(
                                    outputs         = Dz_Dy_Dx_Dt_Db_U,
                                    inputs          = Coords,
                                    grad_outputs    = torch.ones_like(Dz_Dy_Dx_Dt_Db_U),
                                    retain_graph    = True,
                                    create_graph    = True)[0];

            # Update Dz_Dy_Dx_Dt_Db_U (this replaces D_y^{k - 1} Dy_Dx_Dt_Db_U with
            # D_y^k Dy_Dx_Dt_Db_U)
            Dz_Dy_Dx_Dt_Db_U = Grad_Dz_Dy_Dx_Dt_Db_U[:, 3].view(-1);

    return Dz_Dy_Dx_Dt_Db_U;
