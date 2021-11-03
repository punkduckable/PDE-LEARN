import numpy;
import torch;
import random;



def Generate_Points(
        Bounds     : numpy.array,
        Num_Points : int,
        Device     : torch.device = torch.device('cpu')) -> torch.Tensor:

    # First, determine the number of dimensions. This is just the number of rows
    # in Bounds.
    Num_Dim : int = Bounds.shape[0];

    # Check that the Bounds are valid.
    for j in range(Num_Dim):
        assert(Bounds[j, 0] <= Bounds[j, 1]);

    # Make a tensor to hold all the points.
    Points = torch.empty((Num_Points, Num_Dim), dtype = numpy.int64);

    # Populate the coordinates in Points, one coordinate at a time.
    for j in range(Num_Dim):
        # Get the upper and lower bounds for the jth coordinate.
        Lower_Bound : float = Bounds[j, 0];
        Upper_Bound : float = Bounds[j, 1];

        # Cycle through the points.
        for i in range(Num_Points):
            Points[i, j] = random.uniform(Lower_Bound, Upper_Bound);

    return Points;
