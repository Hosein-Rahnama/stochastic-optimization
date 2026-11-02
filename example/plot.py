# This script creates plots of experiments. It should be executed from the the script folder via run.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use LaTeX rendering for plots.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros"],
    "font.size": 10,

    "lines.marker": "o",
    "lines.markersize": 3,
    "lines.linestyle": "-",
    "lines.linewidth": 0.75,

    "axes.labelsize": 10,

    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",

    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 0.5,
    "legend.loc": "upper right"
})

# Set maximum number of epoch to be plotted.
MAX_EPOCH = 25

def parse_experiments(alg, loss, data_set):
    file_name = f"{EXM_DIR}/result/{data_set}/{alg}--{loss}--{data_set}.txt"
    with open(file_name, "r") as f:
        lines = f.readlines()

    experiments = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Experiment"):
            num = int(lines[i].strip().split()[1])
            i += 2
            meta_data = {"number": num, "algorithm": alg, "loss": loss, "data-set": data_set}

            # Parse parameters.
            parameter_header = lines[i].strip().split()
            i += 1
            parameter_values = lines[i].strip().split()
            parameter_values = [float(x) for x in parameter_values]
            parameter = dict(zip(parameter_header, parameter_values))
            i += 2

            # Parse table header.
            table_header = lines[i].strip().split()
            i += 1

            # Parse table rows.
            table = []
            while i < len(lines) and lines[i].strip():
                row = [float(x) for x in lines[i].strip().split()]
                table.append(row)
                i += 1

            experiments.append({
                "meta-data": meta_data,
                "parameter": parameter,
                "result": pd.DataFrame(table, columns=table_header)
            })
        i += 1
    return experiments

def approx_optimal_value(sdca):
    df = sdca["result"]
    optimal_value = (df["primal"].iloc[-1] + df["dual"].iloc[-1]) / 2.0
    return optimal_value

def calc_subopt(experiment, optimal_value):
    df = experiment["result"]
    df["primal-subopt"] = df["primal"] - optimal_value
    if experiment["meta-data"]["algorithm"] != "SGD":
        df["dual-subopt"] = optimal_value - df["dual"]

def calc_sdca_bound(experiment, flag, epoch=0):
    if epoch <= 0:
        epoch = len(experiment["result"])

    lam = experiment["parameter"]["lambda"]
    gamma = experiment["parameter"]["gamma"]
    data_set = experiment["meta-data"]["data-set"]
    info = DATA_SET_INFO[data_set]
    m = info["training-samples"]
    r = np.sqrt(info["maximum-squared-norm"])

    # This constant depends on the loss function and should be modified.
    e = 1.0

    if flag == "smooth":
        beta = 1.0 / gamma
        b = beta * (r ** 2) / (2 * lam)
        c = e * np.exp(1.0 / (m + b))
        values = [c * np.exp(- m * k / (m + b)) for k in range(1, epoch + 1)]
    elif flag == "lipschitz":
        rho = r
        g = (rho ** 2) * (r ** 2) / lam
        c = e + 2.0 * g / m
        values = [c * 1 / (1 + (k / 2)) for k in range(1, epoch + 1)]
    return values

def plot_convergence(sdca, flag, max_epoch = MAX_EPOCH):
    optimal_value = approx_optimal_value(sdca)
    calc_subopt(sdca, optimal_value)
    df = sdca["result"]
    bound = calc_sdca_bound(sdca, flag)

    plt.plot(df["epoch"][0:max_epoch], df["primal-subopt"][0:max_epoch], label="Primal Suboptimality")
    plt.plot(df["epoch"][0:max_epoch], df["dual-subopt"][0:max_epoch], label="Dual Suboptimality")
    plt.plot(df["epoch"][0:max_epoch], df["gap"][0:max_epoch], label="Duality Gap")
    plt.plot(df["epoch"][0:max_epoch], bound[0:max_epoch], label="Bound")

    plt.yscale("log")
    plt.grid(True, which="both", zorder=0)
    plt.legend()

    alg = sdca["meta-data"]["algorithm"]
    lam = sdca["parameter"]["lambda"]
    gamma = sdca["parameter"]["gamma"]
    plt.savefig(f"{EXM_DIR}/plot/{DATA_SET}/convergence--{alg}--{LOSS}--{DATA_SET}--lambda={lam:.0e}--gamma={gamma:g}.pdf", bbox_inches="tight")
    plt.clf()

def plot_smoothness(gamma_low, gamma_mid, gamma_high, max_epoch = MAX_EPOCH):
    df_low = gamma_low["result"]
    df_mid = gamma_mid["result"]
    df_high = gamma_high["result"]

    plt.plot(df_low["epoch"][0:max_epoch], df_low["gap"][0:max_epoch], label="$\\gamma = 0.0$")
    plt.plot(df_mid["epoch"][0:max_epoch], df_mid["gap"][0:max_epoch], label="$\\gamma = 0.1$")
    plt.plot(df_high["epoch"][0:max_epoch], df_high["gap"][0:max_epoch], label="$\\gamma = 1.0$")

    plt.yscale("log")
    plt.grid(True, which="both", zorder=0)
    plt.legend()

    alg = gamma_mid["meta-data"]["algorithm"]
    lam = gamma_mid["parameter"]["lambda"]
    plt.savefig(f"{EXM_DIR}/plot/{DATA_SET}/smoothness--{alg}--{LOSS}--{DATA_SET}--lambda={lam:.0e}.pdf", bbox_inches="tight")
    plt.clf()

def plot_sdca_variants(sdca, sdca_perm, sdca_cyclic, max_epoch = MAX_EPOCH):
    K = max(len(sdca["result"]["epoch"]), len(sdca_perm["result"]["epoch"]), len(sdca_cyclic["result"]["epoch"]))
    epochs = [i for i in range(1, K + 1)]
    bound = calc_sdca_bound(sdca, "smooth", K)

    plt.plot(sdca["result"]["epoch"][0:max_epoch], sdca["result"]["gap"][0:max_epoch], label="SDCA")
    plt.plot(sdca_perm["result"]["epoch"][0:max_epoch], sdca_perm["result"]["gap"][0:max_epoch], label="SDCA-perm")
    plt.plot(sdca_cyclic["result"]["epoch"][0:max_epoch], sdca_cyclic["result"]["gap"][0:max_epoch], label="SDCA-cyclic")

    max_epoch = 0
    for i in range(len(bound)):
        if bound[i] - sdca["result"]["gap"].iloc[-1] >= 0:
            max_epoch += 1
        else:
            break
    max_epoch = min(max_epoch, MAX_EPOCH)
    plt.plot(epochs[0:max_epoch], bound[0:max_epoch], label="Bound")

    plt.yscale("log")
    plt.grid(True, which="both", zorder=0)
    plt.legend()

    lam = sdca["parameter"]["lambda"]
    gamma = sdca["parameter"]["gamma"]
    plt.savefig(f"{EXM_DIR}/plot/{DATA_SET}/gap--SDCA-variants--{LOSS}--{DATA_SET}--lambda={lam:.0e}--gamma={gamma:g}.pdf", bbox_inches="tight")
    plt.clf()

def plot_algs_compare_primal(sdca, sdca_perm, sdca_cyclic, sgd, max_epoch = MAX_EPOCH):
    optimal_value = approx_optimal_value(sdca_perm)
    calc_subopt(sdca, optimal_value)
    calc_subopt(sdca_perm, optimal_value)
    calc_subopt(sdca_cyclic, optimal_value)
    calc_subopt(sgd, optimal_value)

    plt.plot(sdca["result"]["epoch"][0:max_epoch], sdca["result"]["primal-subopt"][0:max_epoch], label="SDCA")
    plt.plot(sdca_perm["result"]["epoch"][0:max_epoch], sdca_perm["result"]["primal-subopt"][0:max_epoch], label="SDCA-perm")
    plt.plot(sdca_cyclic["result"]["epoch"][0:max_epoch], sdca_cyclic["result"]["primal-subopt"][0:max_epoch], label="SDCA-cyclic")
    plt.plot(sgd["result"]["epoch"][0:max_epoch], sgd["result"]["primal-subopt"][0:max_epoch], label="SGD")

    plt.xticks([i for i in range(0, 21, 4)])
    plt.yscale("log")
    plt.grid(True, which="both", zorder=0)
    plt.legend()

    lam = sdca["parameter"]["lambda"]
    gamma = sdca["parameter"]["gamma"]
    plt.savefig(f"{EXM_DIR}/plot/{DATA_SET}/primal--all--{LOSS}--{DATA_SET}--lambda={lam:.0e}--gamma={gamma:g}.pdf", bbox_inches="tight")
    plt.clf()

# Considering the smooth hinge loss, a fixed tolerance and number of epochs, we have four degrees of freedom for each experiment. 
# This includes the choice of algorithm, data set, regularization paramter lambda, and smoothness parameter gamma.

if __name__ == "__main__":
    # Set directories path.
    DAT_DIR = "../data"
    EXM_DIR = "../example"

    # Set data set and loss function.
    LOSS = sys.argv[1]
    DATA_SET = sys.argv[2]

    # Gather data set information from the info.txt files in the data folder.
    DATA_SET_INFO = {}
    data_set_list = ["corp-aq"]
    for data_set in data_set_list:
        file_name = f"{DAT_DIR}/{data_set}/info.txt"
        with open(file_name, "r") as f:
            lines = f.readlines()
            headers = lines[0].strip().split()
            values = lines[1].strip().split()
        values = [int(values[0]), int(values[1]), int(values[2]), int(values[3]), float(values[4]), float(values[5])]
        DATA_SET_INFO[data_set] = dict(zip(headers, values))

    # Gather all experiments from the result folder.
    SDCA = parse_experiments("SDCA", LOSS, DATA_SET)
    SDCA_PERM = parse_experiments("SDCA-perm", LOSS, DATA_SET)
    SDCA_CYCLIC = parse_experiments("SDCA-cyclic", LOSS, DATA_SET)
    SGD = parse_experiments("SGD", LOSS, DATA_SET)

    # Plot the primal and dual suboptimalities and duality gap with the corresponding bounds
    # for SDCA with gamma = 0, 1.
    for i in range(3):
        plot_convergence(SDCA[0 + i], "lipschitz")

    for i in range(3):
        plot_convergence(SDCA[6 + i], "smooth")

    # Plot the effect of smoothness parameter gamma = 0, 0.1, 1 on the convergence of duality gap for SDCA.
    for i in range(3):
        plot_smoothness(SDCA[0 + i], SDCA[3 + i], SDCA[6 + i])

    # Plot duality gap for SDCA, SDCA-perm and SDCA-cyclic with gamma = 1.
    for i in range(3):
        plot_sdca_variants(SDCA[6 + i], SDCA_PERM[6 + i], SDCA_CYCLIC[6 + i])

    # Plot the primal suboptimality for SDCA, SDCA-perm, SDCA-cyclic and SGD with gamma = 0, 1.
    for i in range(3):
        plot_algs_compare_primal(SDCA[0 + i], SDCA_PERM[0 + i], SDCA_CYCLIC[0 + i], SGD[0 + i], 20)

    for i in range(3):
        plot_algs_compare_primal(SDCA[6 + i], SDCA_PERM[6 + i], SDCA_CYCLIC[6 + i], SGD[6 + i], 20)
