`PDE-LEARN` is a PDE discovery algorithm that Robert Stephany and Christopher Earls developed at Cornell University. It can identify a wide variety of linear and non-linear Partial Differential Equations (PDEs) directly from noisy and limited data. This `README` explains how to use PDE-LEARN. Throughout this `README,` we assume the reader is familiar with the PDE-LEARN algorithm. For a complete description of PDE-LEARN, see the paper [PDE-LEARN: Using Deep Learning to Discover Partial Differential Equations from Noisy, Limited Data](https://arxiv.org/abs/2212.04971). If you have questions about our approach and can not find the answer in this document or the paper, send your question to the corresponding author at `rrs254@cornell.edu.`

# Library structure
We split the `PDE-LEARN` repository into several sub-directories and files. The most important of these are `Settings.txt` and `Library.txt.` These files control every variable in the `PDE-LEARN` algorithm. Both files are in `PDE-LEARN`'s main directory. `Settings.txt` houses the settings that control how `PDE-LEARN` operates, while `Library.txt` specifies the left and right-hand side terms of the library of candidate terms for the hidden PDE. When `PDE-LEARN` starts, it parses the contents of `Settings.txt` and `Library.txt` and then identifies a hidden PDE using the parsed settings. **Below, we illustrate how to set up these files.** 

In general, you should not modify the underlying source code. If you want to use `PDE-LEARN` in a way that does not appear possible using the settings in `Setting.txt,` you are welcome to modify the code as you feel fit. With that said, the authors will not be able to help you debug any changes you make. If you decide to modify the source code, we advise you to do so with caution.

*Code:* This directory houses `PDE-LEARN`'s source code. All of our code contains detailed comments. If you want to understand how `PDE-LEARN` works "under the hood," we encourage you to look through our source code. If you find a bug (or anything else that seems unusual) in our code, please tell us about it by emailing the corresponding author at `rrs254@cornell.edu.` 

We split `PDE-LEARN`'s source code over several files. We describe each file below.  

`main.py` is the file you should call when you want to run `PDE-LEARN` (see below). This file reads the settings and library, sets up the system response functions, initializes $\xi$, trains everything, and then saves the results. It drives the rest of the code.

`Data.py` loads a saved data set (in the `.npz` file format). See our description of the *Data* sub-directory below for more details.

`Evaluated_Derivatives.py` houses code that can compute $D_a U$ from $D_b U$ where $D_a$ and $D_b$ are partial derivative operators and $U$ is a neural network. This function assumes that $D_b$ is a "child derivative" of $D_a$, which essentially means that every partial derivative that appears in $D_b$ also appears in $D_a$ and has at least the same order. If $D_a$ is a "child derivative" of $D_a$, then it is possible to compute $D_a U$ from $D_b U$ by taking additional partial derivatives of $D_b U$. Our code exploits this fact to improve the algorithm's runtime and avoid calculating the same partial derivatives multiple times.

`Loss.py` houses code that computes the data, collocation, and $L^p$ loss functions. The collocation loss function computes the partial derivatives of the system response functions using the function `Evaluate_Derivatives.` To maximize efficiency, `PDE-LEARN` first computes low-order partial derivatives of the system response functions and then uses these to compute computing higher-order partial derivatives. This approach minimizes the number of computations required to evaluate the library terms.

`Plot.py` houses code to plot loss histories. Each time the user runs `PDE-LEARN,` it produces plots that depict the data, collocation, $L^2$, and total loss of each system response function, along with $L^p$ loss of $\xi$. The $L^2$ loss of a system response function is the square of the $L^2$ norm of that network's parameters. `PDE-LEARN` then saves these plots in the `Figures` directory.

`Points.py` houses a function that generates the random collocation points.

`Test_Train.py` houses two functions: `Training` and `Testing.` The `Training` function evaluates the system response functions on the testing data set and testing collocation points. `Training` then computes the loss functions using points and uses these values to update $\xi$ and each system response function. By contrast, the `Testing` function evaluates the network/library's performance on a data set but does not update the network's parameters or components of $\xi$. Crucially, `Testing` does NOT modify any network parameters; its purpose is to check if the system response functions are over-fitting.

The `Classes` sub-directory houses definitions of the various classes that `PDE-LEARN` uses.

Finally, the `Readers` sub-directory houses code that parses `Settings.txt` and `Library.txt.`

*Data:* `PDE-LEARN` trains each system response function to match a data set. The "DataSet Names" setting specifies the data sets that `PDE-LEARN` trains on. "DataSet Names" should be a list of strings specifying files in the `Data/DataSets` directory. A "DataSet" is a `.npz` file that contains a dictionary with six keys: "Training Inputs," "Training Targets," "Testing Inputs," "Testing Targets," "Bounds," and "Number of Dimensions." Each of these keys refers to a `numpy.ndarray` object (except "Number of Dimensions," which is an integer that specifies the number of spatial dimensions in the inputs within the data set).

In principle, you can split your dataset into a testing and training set. `PDE-LEARN` will update the system response functions using only the training set data. However, `PDE-LEARN` will report the loss on both the training and test sets. You can use this to determine if the solution networks are overfitting the training set. However, `PDE-LEARN` does not need a testing set to operate correctly. If you want to train the system response functions on all your data, set the testing set to some subset of your training set. Even in this case, `PDE-LEARN` will generate a separate set of collocation points during the testing step.

If you want to use `PDE-LEARN` on your data, you must write a program that calls the `Create_Data_Set` function (in `Create_Data_Set.py`) with the appropriate arguments. See that function's doc-string for details. 

Alternatively, you can create a DataSet using one of our `MATLAB` data sets by running `Python3 ./From_MATLAB.py` when your current working directory is `Data.` The `From_MATLAB` file contains five settings: "Data_File_Name," "Num_Spatial_Dimensions," "Noise_Proportion," "Num_Train_Examples," and "Num_Test_Examples." "Data_File_Name" should refer to one of the `.mat` files in the `MATLAB/Data` directory. "Num_Spatial_Dimensions" specifies the number of spatial dimensions in the inputs stored in the `.mat` file. "Noise_Proportion," "Num_Train_Examples," and "Num_Test_Examples" control the level of noise in the data, the number of training data points, and the number of testing data points, respectively.

*Plot:* The `Plot` directory contains code for visualizing the networks that `PDE-LEARN` trains. In particular, it plots the network's predictions over the problem domain. You can use the file `Plot/Settings.txt` to set up these plots. The file has two settings: "Load File Name" and "Mat File Names." The former specifies the name of the save you want to visualize (this is the file that `PDE-LEARN` saves the system response functions to after training). The latter is a list of strings. The $i$th string should be the name of the `.mat` file that houses the noise-free data that made the noisy and limited data set you used to train the $i$th system response function. Critically, the "Load File Name" setting must refer to a file in `Saves.` To plot a saved system response function, set the appropriate settings in `Plot/Settings.txt` and then run `Python3 ./Plot_Solution.py` when your current working directory is `Plot.`

We need the entire noise-free dataset to evaluate the network's predictions. Therefore, `PDE-LEARN` currently only supports plotting for networks trained on a data set derived from one of the `MATLAB` files.

*Figures:* When `PDE-LEARN` makes a figure, it saves that figure to the `Figures` directory. Thus, the loss-history plots (that `PDE-LEARN` makes each time it runs) and the plots that `Plot_Solution.py` makes end up in this directory. 

*Saves:* When `PDE-LEARN` serializes the network and library at the end of training, it saves the network's state, the library, $\xi$, and other relevant information to a file in the `Saves` directory. PDE-LEARN names the save by appending the network and optimizer type onto the end of the DataSet name(s). If you choose to load a file from save, the "Load File Name" setting in either `Settings.txt` or `Plot/Settings.txt` must refer to a file in the `Saves` directory. 

*Test:* This directory contains the test code we used while developing `PDE-LEARN.`

*MATLAB:* This directory contains the `MATLAB` data sets (the `.mat` files in `MATLAB/Data`) and the scripts that create them (the `.m` files in `MATLAB/Scripts`).



# Settings and the Library: #
In general, to use `PDE-LEARN` you ONLY need to modify the contents of the files `Settings.txt` and `Library.txt.` In particular, you should not need to change any of `PDE-LEARN`'s code; the settings and library files control everything. In this section, we discuss how to use both files. Roughly speaking, the `Library.txt` file defines the left and right-hand side terms ($f_0$ and $f_1, ..., f_K$, respectively; see the [paper](https://arxiv.org/abs/2212.04971) for more details), while `Settings.txt` controls everything else. 

**Settings.txt:** First, let's discuss `Settings.txt.` We organized the settings into categories depending on what aspects of `PDE-LEARN` they control. Below is a discussion of each settings category.


*Save, Load Settings:* "Load U Network from Save," "Load Xi, Library from Save," and "Load Optimizer from Save" specify if you want to start training using a pre-saved system response function, library and $\xi$ vector, or optimizer state, respectively. If any of these settings is `true,` you must specify the "Load File Name" setting, whose should be a file in the `Saves` directory. 

Note that if you plan to load the system response functions, $\xi$, or library from an existing state but want to train using a different optimizer, you CAN NOT load the optimizer state. In general, you can only load the optimizer state if the optimizer setting (see below) matches the optimizer you used to make the save.


*Library Settings:* This section contains just one setting: "Library File." Its value should be the name of the library file you want to use to build the library. Note that `PDE-LEARN` ignores this setting if "Load Xi, Library from Save" is set to `true.` Further note that the library file does NOT need to be called `Library.txt.` The library file can be any text file that adheres to the format of the `Library.txt` file included in this library. 


*Network Settings:* These settings control the architecture of the system response function network(s), $U_1, ... , U_S$. Each network has the same architecture. You specify the width of each layer, as well as the activation function. Each $U_k$ then adopts this architecture. Note that `PDE-LEARN` ignores the architecture settings if the "Load U from Save" setting is `true.` The "Hidden Layer Widths" setting should be a list of integers: the $i$th entry of this list specifies the number of neurons in the $i$th hidden layer of each $U_i$. Likewise, the "Hidden Activation Function" function specifies the activation function we apply after each hidden layer. Currently, `PDE-LEARN` supports three activation function types: Rational (or `Rat`), Hyperbolic Tangent (or `Tanh`), and `Sine.` We recommend using Rational (as we used this activation function for every experiment in the paper). 

Finally, "Train on CPU or GPU" specifies if training should happen on a CPU or GPU. You can only train on a GPU if `PyTorch` supports GPU training on your computer's graphics card. Check `PyTorch`'s website for details.  


*Loss Settings:* "p" specifies the hyperparameter `p` in the $L^p$ loss (see the methodology section of the [paper](https://arxiv.org/abs/2212.04971)). Likewise, "Weights" is a dictionary that must have four keys: "Data," "Coll," "Lp," and "L2". The first three specify $w_{Data}$, $w_{Coll}$, and $w_{L^p}$ (See the methodology section of the [paper](https://arxiv.org/abs/2212.04971)), respectively. Finally, if the value corresponding to "L2" is $c \neq 0$, we add $c$ times the square of the $L^2$ norm of each system response function's parameters to the loss function. The $L^2$ norm acts as a regularizer (it is generally called "weight decay" in the Machine Learning literature). In practice, using a small but non-zero value for the "L2" weight (on the order of $1e-5$) can slightly improve `PDE-LEARN,` though keeping this weight at $0$ generally works fine as well. 

The "Number of Training Collocation Points" and "Number of Testing Collocation Points" settings control the number of RANDOM testing and training collocation points, respectively. Recall that `PDE-LEARN` uses two different kinds of collocation points: Random and targeted. `PDE-LEARN` re-selects the random collocation points at the start of each epoch and selects the targeted ones based on where the PDE residual is largest (see the methodology section of the [paper](https://arxiv.org/abs/2212.04971)). 

Finally, if "Mask Small Xi Components" is `true,` `PDE-LEARN` will stop learning all components of $\xi$ whose initial magnitude is smaller than $0.0005$ (we discuss the reasoning behind this value in the [paper](https://arxiv.org/abs/2212.04971)). Note that `PDE-LEARN` ignores this setting unless you are loading $\xi$ from a save (if "Load Xi, Library from Save" is `true`). 


*Optimizer Settings:* These settings control how `PDE-LEARN` trains $\xi$ and the system response function networks. The "Optimizer" setting specifies which optimizer to train the networks. `PDE-LEARN` supports two optimizers: `Adam` and `LBFGS.` Note that we used the `Adam` optimizer in all of our experiments in the [paper](https://arxiv.org/abs/2212.04971). The "Number of Epochs" and "Learning Rate" settings specify the number of epochs and the optimizer learning rate, respectively. 


*Data settings:* These settings specify where `PDE-LEARN` gets the data it uses to train the system response functions. The "DataSet Names" setting should be a comma-separated list of strings. The ith string should specify the name of a `DataSet` file. See the `Data` section above to understand how to create DataSet files. `PDE-LEARN` makes one system response function per entry in this list. Critically, `PDE-LEARN` saves the data set names when it saves the networks. Thus, if you load the system response function networks from a save, `PDE-LEARN` will ignore this setting. 


**Library.txt:** Now that we know how to set up the Settings, let's discuss `Library.txt.` You must specify two settings in the Library file: The left-hand side term and the right-hand side term. To specify the right-hand side terms, place one term per line. The `Library.txt` file that comes with this repository includes details on how to format a particular library term. Please see that file and the enclosed instructions when setting up your library file. Finally, note that the library file does NOT need to be named `Library.txt.` It can be any text file that adheres to the format specified in the `Library.txt` file included in with repository.


# Running the code: #
Once you have selected the appropriate settings, you can run the code by entering the `Code` directory (`cd ./Code`) and running the main file (`Python3 ./main.py`).

**Burn In:** The first step is the *burn-in* step. For this step, set all of the "load" settings to `false.` Next, select your library and network architecture. For the loss settings, set the "Data" and "Coll" weights to $1.0$ and the "Lp" weight to $0.0$. Make sure that "Mask Small Xi Components" is `true.` Note that this setting will not do anything until the later stages. For the *burn-in* step, we recommend training for $1,000$ epochs using the `Adam` optimizer with a learning rate of $.001$. Select the data sets you want to train on and run the code. Make sure to watch the *data loss* during this stage. If the *data loss* appears to stop decreasing after a few hundred epochs, consider re-running this stage with fewer epochs. In general, letting `PDE-LEARN` train the system response functions after the *data loss* plateaus (stops decreasing) encourages over-fitting and can reduce the accuracy of the final identified PDE. In our experience, the *data loss* stops dropping after $\approx 600-800$ epochs, though it can take more or less depending on the data sets. If the *data loss* is decreasing reasonably quickly after $1,000$ epochs, you can continue training by loading from the save that `PDE-LEARN` made after the first $1,000$ epochs. We suggest training for a few hundred more epochs and then checking if the *data loss* is still decreasing. If so, continue training for more epochs, always loading from the most recent save. Once the *data loss* plateaus, you have finished the *burn-in* step.

**Sparseification:** The second step is the *sparseification* step. For this step, set all of the "load" settings to `true.` Set the "Load File Name" setting to the name of the save from the end of the *burn-in* step. Change the "Lp" weight to a small, positive value like $0.0002$. Otherwise, you should use the same settings that you used in the *burn-in* stage (note that `PDE-LEARN` will ignore any changes you make to the architecture and or data settings). We recommend training for another $1,000$ epochs using the `Adam` optimizer with a learning rate of $0.001$. Run the code and watch the *Lp loss* as it runs. If the *Lp loss* has not decreased significantly in $\approx 200$ epochs, you can probably stop training. Usually, this takes around $1,000$ epochs, though it sometimes takes more. After training, look at the "*Lp loss* history" plot. The plot should look like a staircase (with each step corresponding to one of the components of $\xi$ dropping to zero). If you think the *Lp loss* might drop down more "steps," you can run the *sparsification* step for additional epochs (loading from the save produced at the end of the first $1,000$ epochs of training). Once the *Lp loss* stabilizes, you have finished the *sparsification* step. 

**Fine-tuning:** The fine-tuning step is the final stage. For this step, keep all of the "load" settings to `true,` but change the "Load File Name" setting to the name of the save from the end of the *sparsification* step. Change the "Lp" weight to $0.0$. Otherwise, we recommend using the same settings from the *sparsification* stage. We recommend training for $1,000$ epochs using the `Adam` optimizer using a learning rate of $.001$. However, the optimal number of epochs in this step depends significantly on the underlying data set. In some experiments, we used $50$ or fewer fine-tuning epochs. In other experiments, we used several thousand. When this step runs, closely watch the *Lp loss*. Almost always, a pattern emerges: the *Lp loss* increases for a while, plateaus, and then decreases. You want to stop training when the *Lp loss* plateaus. One way to do this is to train for a few thousand epochs. Once you see the *Lp loss* plateau, record the number of epochs and then kill `PDE-LEARN` (by pressing `Ctrl + C`). Set the number of epochs to the number you wrote down and re-run `PDE-LEARN.` This approach usually yields the most accurate constants. We want to emphasize, however, that the *fine-tuning* step does NOT change the FORM of the identified PDE, only the constants in it. Stopping training before or after the "plateau" in the *Lp loss* will not change the form of the identified PDE but decreases the accuracy of its constants.

**What to do if you get nan:** `PDE-LEARN` can use the `LBFGS` optimizer. Unfortunately, `PyTorch`'s `LBFGS` optimizer is known to yield nan (see <https://github.com/pytorch/pytorch/issues/5953>). Using the `LBFGS` optimizer occasionally causes `PDE-LEARN` to break down and start reporting `nan.` If this happens, you should kill `PDE-LEARN` (in the terminal window, press `Ctrl + C`) and then re-run `PDE-LEARN.` Since `PDE-LEARN` randomly samples the collocation points from the problem domain, no two runs of `PDE-LEARN` are identical. Thus, even if you keep the settings the same, re-running `PDE-LEARN` may avoid the `nan` issue. If you encounter `nan` on several successive runs of `PDE-LEARN,` reduce the learning rate by a factor of $10$ and try again. If all else fails, consider training using another optimizer.


# Dependencies: #
`PDE-LEARN` will not run unless you have installed the following:

* `Python3`
* `numpy`
* `torch`
* `matplotlib`
* `pandas`
* `seaborn`

Additionally, you'll need `scipy` if you want to use the `From_MATLAB.py` function in the `Data` directory.