import torch;

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
