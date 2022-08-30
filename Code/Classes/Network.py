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
        """ 
        This function applies a rational function to each element of X.

        ------------------------------------------------------------------------
        Arguments:

        X: A tensor. We apply the rational function to every element of X.

        ------------------------------------------------------------------------
        Returns:

        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. 
        """

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
                    Hidden_Activation   : str,
                    Output_Activation   : str           = "None",
                    Device              : torch.device  = torch.device('cpu')):
        """
        This is the initializer for the Network class.
        
        -----------------------------------------------------------------------
        Arguments: 

        Widths: This is a list of integers whose ith entry specifies the width
        of the ith layer of the network, including the input and output layers
        (the input layer is the 0 layer).

        Hidden_Activation: The activation function we use after each hidden 
        layer.

        Output_Activation: The activation function we on the output layer. 
        Default is "None".

        Device: The device we want to load the network on.
        """
        
        for i in range(len(Widths)):
            assert(Widths[i] > 0), ("Layer widths must be positive got Layer[%u] = %d" % (i, Widths[i]));
    
        super(Network, self).__init__();

        # Define object attributes. Note that we only count layers with 
        # trainable parameters (the hidden and output layers). Thus, this does
        # not include the input layer.        
        self.Widths             : List[int] = Widths; 
        self.Num_Layers         : int       = len(Widths) - 1;
        self.Num_Hidden_Layers  : int       = self.Num_Layers - 1;


        #######################################################################
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

        # Initialize the weight matrices, bias vectors.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(  self.Layers[i].weight);
            torch.nn.init.zeros_(           self.Layers[i].bias);


        #######################################################################
        # Set up Activation Functions

        self.Activation_Functions = torch.nn.ModuleList();
        for i in range(self.Num_Hidden_Layers):
            self.Activation_Functions.append(self._Get_Activation_Function(Encoding = Hidden_Activation, Device = Device));

        self.Activation_Functions.append(self._Get_Activation_Function(Encoding = Output_Activation, Device = Device));



    def _Get_Activation_Function(self, Encoding : str, Device : torch.device) -> torch.nn.Module:
        """
        An internal function which converts a string into its corresponding 
        activation function.

        -----------------------------------------------------------------------
        Arguments:

        Encoding: A string which specifies an activation function. Currently, 
        the following Encodings are allowed:
            "None"
            "Elu"
            "Tanh"
            "Sigmoid"
            "Rational"
            "Softmax"
        
        -----------------------------------------------------------------------
        Returns:

        The corresponding activation function object (all activation classes
        are sub-classes of torch.nn.Module).
        """

        Processed_Encoding : str = Encoding.strip().lower();

        if(  Processed_Encoding == "none"):
            return torch.nn.Identity();
        elif(Processed_Encoding == "tanh"):
            return torch.nn.Tanh();
        elif(Processed_Encoding == "sigmoid"):
            return torch.nn.Sigmoid();
        elif(Processed_Encoding == "elu"):
            return torch.nn.ELU();
        elif(Processed_Encoding == "softmax"):
            return torch.nn.Softmax();
        elif(Processed_Encoding == "rational"):
            return Rational(Device = Device);
        else:
            print("Unknown Activation Function. Got %s" % Encoding);
            exit();



    def _Get_Activation_String(self, Activation : torch.nn.Module) -> str:
        """
        An internal function which converts an activation function into a
        corresponding string. This function is the inverse of 
        _Get_Activation_Function.

        -----------------------------------------------------------------------
        Arguments:

        Activation: An instance of an activation function class.
        
        -----------------------------------------------------------------------
        Returns:

        A string describing the type of activation function. For example, if 
        Activation is a Tanh object, then the returned string is "Tanh".
        """

        if(  isinstance(Activation, torch.nn.Identity)):
            return "None";
        elif(isinstance(Activation, torch.nn.Tanh)):
            return "Tanh";
        elif(isinstance(Activation, torch.nn.Sigmoid)):
            return "Sigmoid";
        elif(isinstance(Activation, torch.nn.ELU)):
            return "Elu";
        elif(isinstance(Activation, torch.nn.Softmax)):
            return "Softmax";
        elif(isinstance(Activation, Rational)):
            return "Rational";
        else:
            print("Unknown Activation Function. Got %s" % str(type(Activation)));
            exit();



    def Get_State(self) -> Dict:
        """
        This function returns a dictionary that houses everything you'd need 
        to recreate this object from scratch. The returned dictionary is 
        essentially a "super" state dictionary that goes far beyond what's 
        contained in a regular state dictionary. This dictionary can be 
        passed to the Get_State method of another object to create an identical 
        copy of self. 

        -----------------------------------------------------------------------
        Returns:

        A dictionary which can be used to recreate self from scratch.
        """

        # Initialize the State dictionary with the Widths attribute.
        State : Dict =  {   "Widths"    : self.Widths };

        # Make a list whose ith entry holds the ith layer's state dict. Then 
        # add this list to State.
        Layer_State_Dicts : List[Dict] = [];
        for i in range(self.Num_Layers):
            Layer_State_Dicts.append(self.Layers[i].state_dict());

        State["Layers"] = Layer_State_Dicts;

        # Make two lists whose ith entries hold the activation string and 
        # activation state for the ith hidden activation function, respectively. 
        # Note that if the ith activation function has no internal state, 
        # which is the case for all activation functions other than Rational,
        # then its state dict is empty. Thankfully, passing an empty state 
        # dict to the load_state_dict method of such an object does nothing,
        # so we do not need to handle activation functions with an internal
        # state differently from those without one.
        Activation_Types    : List[str]     = [];
        Activation_States   : List[dict]    = [];
        for i in range(self.Num_Layers):
            Activation_Types.append(self._Get_Activation_String(self.Activation_Functions[i]));
            Activation_States.append(self.Activation_Functions[i].state_dict());
        
        State["Activation Types"]    = Activation_Types;
        State["Activation States"]   = Activation_States;

        # All done!
        return State;



    def Set_State(self, State : Dict) -> None:
        """
        This function sets self's state according to the State argument. State
        should be a state dictionary returned by the "Get_State" method (or 
        an un-pickled version of one). We assume that the passed State is 
        compatible with self in the sense that they use the same architecture 
        (Widths and activation functions).
        """

        # Check for compatibility
        assert(len(self.Widths) == len(State["Widths"]));
        for i in range(len(self.Widths)):
            assert(self.Widths[i] == State["Widths"][i]);
        
        for i in range(self.Num_Layers):
            assert(self._Get_Activation_String(self.Activation_Functions[i]) == State["Activation Types"][i]);
        
        # Now update self's state!
        for i in range(self.Num_Layers):
            self.Layers[i].load_state_dict(State["Layers"][i]);
        
        for i in range(self.Num_Layers):
            self.Activation_Functions[i].load_state_dict(State["Activation States"][i]);
        
        

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ 
        Forward method for the NN class. Note that the user should NOT call
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
        applied to the ith row of X.
        """

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (which has no activation function) and
        # return.
        return self.Layers[self.Num_Layers - 1](X);
