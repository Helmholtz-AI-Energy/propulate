import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

functions = ("Sphere", "Rosenbrock", "Step", "Quartic", "Griewank", "Rastrigin", "Schwefel", "BiSphere", "BiRastrigin")
function_name = functions[8]

# Nötige Nacharbeiten:
# ?

if __name__ == "__main__":
    path = Path(".")
    data = []
    # pso_names = ("VelocityClamping", "Constriction", "Basic", "Canonical")
    marker_list = ("o", "s", "D", "^") # ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]
    for p in path.glob(f"bm_H_{function_name.lower()}_?"):
        if not p.is_dir():
            continue
        data.append([])
        for file in p.iterdir():
            if not file.suffix == ".pkl":
                continue
            with open(file, "rb") as f:
                data[-1].append(pickle.load(f, fix_imports=True))
        if len(data[-1]) == 0:
            del data[-1]
    for i, _ in enumerate(data):
        data[i] = [dx for dx in data[i] if not any([dxt is None for dxt in dx])][0][0]
    del _

    plt_data = []
    for x in data:
        entry = [min(x["losses"]), 2000 / len(x)]
        plt_data.append(entry)
    plt_data = np.array(sorted(plt_data, key=lambda v: v[1])).T

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots()
    # fig.subplots_adjust(hspace=0)

    ax.set_title(f"Vanilla Propulate on {function_name} function")
    ax.set_xlabel("Nodes")

    ax.plot(plt_data[1], plt_data[0], label="Hyppopy", marker="P", ms=10, ls="dotted", lw=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    ax.grid(True)
    ax.set_ylabel("Loss")
    if function_name in ("Step", "Griewank", "Rastrigin", "Schwefel", "BiRastrigin"):
        ax.set_yscale("linear")
    else:
        ax.set_yscale("log")
    ax.legend()

    fig.show()

    # save_path = Path(f"images/propulate/{function_name.lower()}.png")
    # if save_path.parent.exists() and not save_path.parent.is_dir():
    #     OSError("There is something in the way. We can't store our paintings.")
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    #
    # fig.savefig(save_path)
    # fig.savefig(save_path.with_suffix(".svg"))
    # fig.savefig(save_path.with_suffix(".pdf"))
    # fig.savefig(save_path.with_stem(save_path.stem + "_T"), transparent=True)
    # fig.savefig(save_path.with_stem(save_path.stem + "_T").with_suffix(".svg"), transparent=True)
    # fig.savefig(save_path.with_stem(save_path.stem + "_T").with_suffix(".pdf"), transparent=True)
