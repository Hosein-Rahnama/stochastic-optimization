# Content

This library implements some versions of Stochastic Gradient Decent (SGD) and Stochastic Dual Coordinate Ascent (SDCA) algorithms used for optimization in machine learning. For more details on SDCA, see [this work](https://www.jmlr.org/papers/v14/shalev-shwartz13a.html).


# Quick Start

In the following series of commands, I first clean everything including the results of the previous runs and the corresponding plots. Next, I run all of the algorithms for `corp-aq` dataset with `hinge-loss`. Finally, I plot the results.

```bash
./run clean --all
./run make
./run all hinge-loss corp-aq
./run plot hinge-loss corp-aq
```


# User Guide

To use the library, the user has three main tasks:
1.  Implement the desired loss function.
2.  Prepare the desired dataset in **SVM-light** format.
3.  Specify the numerical-experiment settings in `experiment.txt`.

Regarding the implementation of the loss function, the following four functions must be implemented. They are declared in the header file `run.hpp`. For example, these functions have been implemented for the smooth hinge loss in `hinge-loss.cpp`. Therefore, to implement the squared loss, it is sufficient to place a file named `square.cpp`, containing implementations of the functions below, in the `example` directory. When executing the commands that follow, assume that the current directory is `script`.

``` cpp
// Compute primal objective function according to the loss function.
double primal(DataSet& trainData, SimpleVector& w, double lambda, double gamma);

// Compute dual objective function according to conjugate of the loss function.
double dual(DataSet& trainData, SimpleVector& w, SimpleVector& a,
            double lambda, double gamma);

// Compute step size for SDCA according to the loss function.
double step(DataSet& trainData, SimpleVector& w, SimpleVector& a, uint i,
            double lambda, double gamma);

// Compute subgradient of the loss function for SGD.
double subgradient(DataSet& trainData, SimpleVector& w, uint i, double gamma);
```

Each new dataset must be prepared in **SVM-light** format. In the `data` directory, create a subdirectory named after the dataset. This subdirectory must contain the files `train.dat` and `test.dat`.

To extract the information about the new dataset, run the following command, where `<data-set>` is the name of the dataset. After this command has been executed, an `info.txt` file will be created in the corresponding directory.

``` sh
./run data <data-set>
```

For the numerical-experiment settings, refer to the `experiment.txt` file. The first line indicates the number of experiments. Each subsequent line contains four numbers separated by spaces:

1.  The first number is the parameter is the regularization parameter $\lambda$.
2.  The second number is the parameter $\gamma$, which is the inverse of the smoothness constant $\beta$ of the loss function for a specific sample.
3.  The third number is the tolerance required for the duality gap.
4.  The fourth number is the maximum number of training epochs.

Now, to run the numerical experiments, execute the following command:

``` sh
./run <algorithm> <loss> <data-set>
```

In this command, `<algorithm>` is one of the following algorithm names, `SDCA`, `SDCA-perm`, `SDCA-cyclic`, or `SGD`. `<loss>` is the name of the file in which the implementations associated with the loss function have been placed. `<data-set>` is the name of the desired dataset.

After processing is complete, the results of the numerical experiments
are saved in the `result` directory.

If you want to run all the algorithms, you can use the following
command:

``` sh
./run all <loss> <data-set>
```