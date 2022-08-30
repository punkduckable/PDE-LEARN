import  torch;
import  math;
from    typing  import Dict, List;



class Rational(torch.nn.Module):
    def __init__(self,
                 Device    = torch.device('cpu')):
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__();

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype  = torch.float32,
                                     device = Device));
        self.a.requires_grad_(True);

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype  = torch.float32,
                                     device = Device));
        self.b.requires_grad_(True);


    def forward(self, X : torch.tensor):
        """ This function applies a rational function to each element of X.

        ------------------------------------------------------------------------
        Arguments:

        X: A tensor. We apply the rational function to every element of X.

        ------------------------------------------------------------------------
        Returns:

        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a;
        b = self.b;

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X));
        D_X = b[0] + X*(b[1] + b[2]*X);

        # Return R = N_X/D_X. This is also applied element-wise.
        return torch.div(N_X, D_X);



class Network(torch.nn.Module):
    def __init__(   self,
                    Widths              : List[int],
                    Device              : torch.device = torch.device('cpu'),
                    Activation_Function : str          = "Tanh"):
        """
        This is the initializer for the Network class.
        
        -----------------------------------------------------------------------
        Arguments: 

        Widths: This is a list of integers whose ith entry specifies the width
        of the ith layer of the network, including the input and output layers
        (the input layer is the 0 layer).

        Device: The device we want to load the network on.

        Activation_Function: The activation function we use after each hidden 
        layer.
        """
        
        for i in range(len(Widths)):
            assert(Widths[i] > 0), ("Layer widths must be positive got Layer[%u] = %d" % (i, Widths[i]));
    
        super(Network, self).__init__();

        # Define object attributes. Note that we only count layers with 
        # trainable parameters (the hidden and output layers). Thus, this does
        # not include the input layer.
        self.Input_Dim      : int = Widths[0];
        self.Output_Dim     : int = Widths[-1];
        self.Num_Layers     : int = len(Widths) - 1;

        Num_Hidden_Layers   : int = self.Num_Layers - 1;


        ########################################################################
        # Set up the Layers

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the layers! The domain of the ith layer is R^Widths[i] and its 
        # co-domain is R^Widths[i + 1]. Thus, the domain of the '0' layer is 
        # the input layer while its co-domain is the first hidden layer.
        for i in range(self.Num_Layers):
            self.Layers.append(torch.nn.Linear( 
                                    in_features     = Widths[i],
                                    out_features    = Widths[i + 1],
                                    bias            = True).to(dtype = torch.float32, device = Device));


        ########################################################################
        # Set up Activation Functions

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList();
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh());
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational(Device = Device));
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function);
            print("Thrown by Network.__init__. Aborting.");
            exit();


        ########################################################################
        # Initialize the weight matrices, bias vectors.

        # If we're using Rational or Tanh networks, use xavier initialization.
        if(Activation_Function == "Tanh" or Activation_Function == "Rational"):
            for i in range(self.Num_Layers):
                torch.nn.init.xavier_uniform_(  self.Layers[i].weight);
                torch.nn.init.zeros_(           self.Layers[i].bias);



    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class. Note that the user should NOT call
        this function directly. Rather, they should call it through the __call__
        method (using the NN object like a function), which is part of the
        module class and calls forward.

        ------------------------------------------------------------------------
        Arguments:

        X: A batch of inputs. This should be a B by Input_Dim tensor, where B
        is the batch size. The ith row of X should hold the ith input.

        ------------------------------------------------------------------------
        Returns:

        If X is a B by Input_Dim tensor, then the output of this function is a
        B by Output_Dim tensor, whose ith row holds the value of the network
        applied to the ith row of X. """

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Layers - 1):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (which has no activation function) and
        # return.
        return self.Layers[self.Num_Layers - 1](X);
