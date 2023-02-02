`PDE-LEARN` is a PDE discovery algorithm that was developed by Robert Stephany and Christopher Earls at Cornell University. It can identify a wide variety of linear and non-linear Partial Differential Equations (PDEs) directly from noisy and limited data. This `READ-ME` explains how to use `PDE-LEARN`. Throughout this document, we assume the reader is familiar with the `PDE-LEARN` algorithm. For a full description of the algorithm, please see the paper *PDE-LEARN: Using Deep Learning to Discover Partial Differential Equations from Noisy, Limited Data*.  If you have a questions about our approach and can not find the answer in this document or the paper, either send your question to the correspond author at *rrs254@cornell.edu*, or consult the code directly (all of which is thoroughly commented).

# Library structure
The `PDE-LEARN` repository is split into several sub-directories and files. For most users, the most important of these are the files `Settings.txt` and `Library.txt`, both of which are located in the repository's main directory. `Settings.txt` houses the settings that control how `PDE-LEARN` operates, while `Library.txt` specifies the left and right-hand side terms. When `PDE-LEARN` runs, it parses the contents of `Settings.txt` and `Library.txt` and then uses the values it reads to identify a PDE. **Below, we how to set up these files.** These two files completely control `PDE-LEARN`. As a result, you should not modify the underlying source code. If you want to use `PDE-LEARN` in a way that does not appear possible using the settings in `Setting.txt`, you are welcome to modify the code as you feel fit; though the authors will not be able to help you debug any changes you make.

*Code:*  This directory houses `PDE-LEARN`'s source code. All of our code is fully commented. Feel free to look through our code to understand how `PDE-LEARN` works "under the hood".  If you think you've found a bug in our code, please let us know! `PDE-LEARN`'s source code is split over several files. 

The `main.py` file is the one that you should call when you want to run `PDE-LEARN` (see below). This file reads the settings and library, sets up the networks, trains them, and then saves the results. It drives the rest of the code.

The file `Data.py` loads a saved data set (in the `.npz` file format, see the *Data:* section below for more details).

The `Evaluated_Derivatives.py` file houses code that can compute $D_a U$ from $D_b U$ where $D_a$ and $D_b$ are partial derivative operators and $U$ is a neural network. This function assumes that $D_b$ is a "child derivative" of $D_a$, which essentially means that every partial derivative that appears in $D_b$ also appear in $D_a$, and have at least the same order. If $D_a$ is "child derivative" of $D_a$, then it is possible to compute $D_a U$ from $D_b U$ by applying additional partial derivatives to $D_b U$. 

The `Loss.py` file houses code that computes the data, collocation, and $L^p$ losses. The collocation loss function computes the partial derivatives of $U$ using the `Evaluate_Derivatives` function. It does this by first evaluating the lowest order partial derivatives of $U$, and then computing higher order partial derivatives of $U$ from these. This approach minimizes the number of computations required to evaluate the library terms.

The `Plot.py` function houses code to plot loss histories. Each time that the user runs `PDE-LEARN`, it produces plots that depict the loss history of the three loss functions of $U$, as well as the total $L^2$ loss of $U$'s parameters, and the total loss. These plots are saved into the `Figures` directory.

`Points.py` houses a function that generates the random collocation points.

Finally, `Test_Train.py` houses two functions: `Testing` and `Training`. The former evaluates the network/library's performance on a tata set, but does not update the network's parameters or components of $\xi$. The `Training` function implements on epoch of training.

The `Classes` sub-directory houses definitions of the various classes that `PDE-LEARN` uses.

Finally, the `Readers` sub-directory houses code that parses `Settings.txt` and `Library.txt`.

*Data:* `PDE-LEARN` trains $U$ to match a data set. The "DataSet Names" setting specifies the data sets that `PDE-LEARN` trains on. "DataSet Names" should be a list of strings specifying files that are in the folder `Data/DataSets` directory. A "DataSet" is simply a `.npz` file that contains a dictionary with six keys: "Training Inputs", "Training Targets", "Testing Inputs", "Testing Targets", "Bounds" and "Number of Dimensions." Each of these keys refers to a `numpy.ndarray` object (except "Number of Dimensions", which is just an integer specifying the number of spatial dimensions in the inputs in the data set). If you want to use `PDE-LEARN` on your data, you must write a program that calls the `Create_Data_Set` function (in `Create_Data_Set.py`) with the appropriate arguments. See that function's docstring for details. Alternatively, you can create a DataSet using one of our `MATLAB` data sets by running `Python3 ./From_MATLAB.py` when your current working directory is `Data`. The `From_MATLAB` file contains five settings: "Data_File_Name", "Num_Spatial_Dimensions", "Noise_Proportion", "Num_Train_Examples", and "Num_Test_Examples". "Data_File_Name" should refer to one of the `.mat` files in the `MATLAB/Data` directory. "Num_Spatial_Dimensions" specifies the number of spatial dimensions in the inputs stored in the `.mat` file. "Noise_Proportion", "Num_Train_Examples", and "Num_Test_Examples" control the level of noise in the data, the number of training data points, and the number of testing data points, respectively.

*Plot:* Visualizing $U$ after `PDE-LEARN` trains it can be very useful. The `Plot` directory contains code for visualizing the networks that `PDE-LEARN` trains. Plotting is controlled by the `Plot/Settings.txt` file. This file has just two settings: "Load File Name" and "Mat File Names." The former specifies the name of the save that you want to visualize (this is the file that `PDE-LEARN` saves the trained $U$ network to after training). The latter specifies the name of the `.mat` file that houses the noise-free data set corresponding to the data set that you trained $U$ trained on. Critically, the "Load File Name" setting must refer to a file in `Saves`. To plot a saved solution, set the appropriate settings in `Plot/Settings.txt` and then run `Python3 ./Plot_Solution.py` when your current working directory is `Plot.`

*Figures:* When `PDE-LEARN` makes a figure, it saves that figure to the `Figures` directory. Thus, the loss-history plots (that `PDE-LEARN` makes each time it runs), as well as the plots that `Plot_Solution.py` makes end up in this directory. 

*Saves:* When `PDE-LEARN` directory serializes the network and library at the end of training, it saves that state to a file in the `Saves` directory. PDE-LEARN uses a specific naming scheme that effectively appends the network type and optimizer type onto the end of the DataSet name (in the `Settings.txt` file). If you choose to load a file from save, the "Load File Name" setting in either `Settings.txt` or `Plot/Settings.txt` must refer to a file in the `Saves` directory. 

*Test:* This directory contains test code that I used while developing `PDE-LEARN`.

*MATLAB:* This directory contains the `MATLAB` data sets (the `.mat` files in `MATLAB/Data`), and the scripts that create them (the `.m` files in `MATLAB/Scripts`).




# Settings and the Library: #
The `Settings.txt` and `Library.txt` files controls basically every aspect of `PDE-LEARN`. In general, to use `PDE-LEARN` you ONLY need to modify the contents of these files. In particular, you should not need to modify any code in the `Code` directory; the settings and library files control everything. In this section, I will discuss how to use both files. Roughly speaking, the `Library.txt` file defines the left and right hand side terms ($f_0$ and $f_1, ... , f_K$, respectively; see the paper for more details), while `Settings.txt` controls everything else. 

First, let's discuss `Settings.txt`. I organized the settings into categories depending on what aspects of `PDE-LEARN` they control. Below is a discussion of each settings category.


*Save, Load Settings:* "Load U Network from Save", "Load Xi, Library from Save", and "Load Optimizer from Save" specify if you want to start training using a pre-saved Solution Network, Library and $\xi$ vector, or Optimizer state, respectively. If any one of these settings is true, you must specify the "Load File Name" setting, which should specify a file in the `Saves` directory. Note that if you plan to load from an existing state, but want to train 
using a different optimizer, you CAN NOT load the optimizer state. In general, you can only loss the optimizer state if the optimizer setting (see below) matches the optimizer that you used to make the save.


*Library Settings:* This section contains just one setting: "Library File". This should be set to the name of the library file you want to build the library from. Note that `PDE-LEARN` ignores this setting if "Load Xi, Library from Save" is set to true. Further note that the library file does NOT need to be called `Library.txt`. It can be any text file that adheres to the format of the `Library.txt` file included in this library. We included this flexibility to allow users to easily use different libraries for different problems. 


*Network Settings:* These settings control the architecture of the solution network(s), $U_1, ... , U_S$. Each network has the same architecture. You specify the width each layer as well as the activation function. Each $U_k$ then adopts this architecture. Note that `PDE-LEARN` ignores the architecture settings if the "Load U from Save" setting is true. The "Hidden Layer Widths" setting should be a list of integers. The $i$th entry of this list specifies the number of neurons the $i$th hidden layer of each $U_i$. Likewise, the "Hidden Activation Function" function specifies the activation function we apply after each hidden layer. Currently, `PDE-LEARN` supports three activation function types: Rational (or `Rat`), Hyperbolic Tangent (or `Tanh`), and `Sine`. We recommend using Rational (as we used this activation function for every experiment in the paper). 

Finally, "Train on CPU or GPU" specifies if training should happen on a CPU or GPU. You can only train on a GPU if `torch` supports GPU training on your computer's graphics card. Check torch's website for details.  


*Loss Settings:* "p" specifies the number `p` in the Collocation loss (see the methodology section of the paper). Likewise, "Weights" is a dictionary that must have four keys: "Data", "Coll", "Lp", and "L2". The first three specify $w_{Data}$, $w_{Coll}$, and $w_{L^p}$ (See the methodology section of the paper), respectively. Finally, if the value corresponding to "L2" is $c \neq 0$, we add on $c$ times the square of the $L^2$ norm of $U's$ parameters to the loss function. This acts as a weak regularizer (it is generally called "weight decay" in the Machine Learning literature). In practice, using a small but non-zero value for the "L2" weight (on the order of $5e-6$) can slightly improve `PDE-LEARN`, though keeping this weight at $0$ generally works fine as well. 

The "Number of Training Collocation Points" and "Number of Testing Collocation Points" settings control the number of RANDOM testing and training collocation points, respectively. Recall that `PDE-LEARN` uses two kinds of collocation points: Random and targeted. `PDE-LEARN` re-selects the random collocation points at the start of each epoch, and selects the Targeted Collocation Points based on where the PDE-residual is largest (see the methodology section of the paper). 

Finally, if "Mask Small Xi Components" is set to true, `PDE-LEARN` will stop training all components of $\xi$ whose magnitude starts of smaller than $0.0005$ (we discuss the reasoning behind this value in the paper). Note that `PDE-LEARN` ignores this setting unless you are loading $\xi$ from a save (if "Load Xi, Library from Save" is set to true). 


*Optimizer Settings:* These settings control how `PDE-LEARN` trains the solution networks and $\xi$. The "Optimizer" setting specifies which optimizer to train the networks. `PDE-LEARN` supports two optimizers: `Adam` and `LBFGS`. Note that we used the `Adam` optimizer in all of our experiments in the paper. 
The "Number of Epochs" and "Learning Rate" settings specify the number of epochs and the learning rate for the optimizer specified in the "Optimizer" setting, respectively. 


*Data settings:* These settings specify where `PDE-LEARN` gets the data it uses to train the networks. The "DataSet Names" setting should be a comma-separated list of strings. The ith string should specify the name of a `DataSet` file. See the `Data` section above to understand how to create DataSet files. `PDE-LEARN` makes one solution network per entry in this list. Crucially, if you save a state from save, the "DataSet Names" must match those you used to make the save. Thus, if you want to load a network that you trained trained on two DataSets, "DataSet Names" should contain the names of the two DataSets (in the same order!). 


*Library.txt:* Now that we know to set up the Settings, let's discuss `Library.txt`. You must specify two settings in the Library file: The left hand side term, and the right hand side term. To specify the right hand side terms, place one term per line. The file `Library.txt` that comes with this repository includes details on how to specify a particular library term. Please see that file and those instructions when setting up your library. Finally, note that the library file does NOT need to be called `Library.txt`. It can be any text file that adheres to the format specified in the `Library.txt` file included in this library.


# Running the code: #
 Once you have selected the appropriate settings, you can run the code by entering the `Code` directory (`cd ./Code`) and running the main file (`Python3 ./main.py`).

 *What to do if you get nan:* `PDE-LEARN` can use the `LBFGS` optimizer. Unfortunately, PyTorch's `LBFGS` optimizer is known to yield nan (see <https://github.com/pytorch/pytorch/issues/5953>). Using the `LBFGS` optimizer occasionally causes `PDE-LEARN` to break down and start reporting nan. If this occurs, you should kill `PDE-LEARN` (in the terminal window, press `Ctrl + C`), and then re-run `PDE-LEARN.` Since `PDE-LEARN` randomly samples the collocation points from the problem domain, no two runs of `PDE-LEARN` are identical. Thus, even if you keep the settings the same, re-running `PDE-LEARN` may avoid the nan issue. If you encounter nan on several successive runs of `PDE-LEARN,` reduce the learning rate by a factor of $10$ and try again. If all else fails, consider training using another optimizer.


# Dependencies: #
`PDE-LEARN` will not run unless you have installed the following:
* `Python3`
* `numpy`
* `torch`
* `matplotlib`
* `pandas`
* `seaborn`

Additionally, you'll need `scipy` if you plan to use the `From_MATLAB.py` function in the `Data` directory.
