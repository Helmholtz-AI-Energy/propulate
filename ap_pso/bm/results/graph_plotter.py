import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

functions = ("Sphere", "Rosenbrock", "Step", "Quartic", "Griewank", "Rastrigin", "Schwefel", "BiSphere", "BiRastrigin")
function_name = functions[8]

if __name__ == "__main__":
    path = Path(".")
    data = []
    pso_names = ("VelocityClamping", "Constriction", "Basic", "Canonical")
    marker_list = ("o", "s", "D", "^", "P") # ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]
    # np.random.shuffle(marker_list)
    for i in range(4):
        data.append([])
        for p in path.glob(f"bm_{i}_{function_name.lower()}_?"):
            if not p.is_dir():
                continue
            for file in p.iterdir():
                if not file.suffix == ".pkl":
                    continue
                with open(file, "rb") as f:
                    data[i].append(pickle.load(f, fix_imports=True))
    data.append([])
    for p in path.glob(f"bm_P_{function_name.lower()}_?"):
        if not p.is_dir():
            continue
        for file in p.iterdir():
            if not file.suffix == ".pkl":
                continue
            with open(file, "rb") as f:
                data[-1].append(pickle.load(f, fix_imports=True))

    plt_data = []
    for i in range(5):
        plt_data.append([])
        for x in data[i]:
            entry = [min(x, key=lambda v: v.loss).loss, max(x, key=lambda v: v.rank).rank + 1]
            plt_data[i].append(entry)
        plt_data[i] = np.array(sorted(plt_data[i], key=lambda v: v[1])).T

    fig: Figure
    ax: Axes

    fig, ax = plt.subplots()
    # fig.subplots_adjust(hspace=0)

    ax.set_title(f"PSO@Propulate on {function_name} function")
    ax.set_xlabel("Nodes")

    for i in range(4):
        ax.plot(plt_data[i][1], plt_data[i][0], label=pso_names[i], marker=marker_list[i], ls="dotted", lw=2)
    ax.plot(plt_data[4][1], plt_data[4][0], label="Vanilla Propulate", marker=marker_list[4], ls="dotted", lw=2, ms=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    ax.grid(True)
    ax.set_ylabel("Loss")
    if function_name == "Rosenbrock":
        ax.set_yscale("symlog", linthresh=1e-19)
        ax.set_yticks([0, 1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1, 100])
        ax.set_ylim(-5e-20, 100)
    elif function_name == "Step":
        ax.set_yscale("symlog")
        ax.set_ylim(-1e4, -5)
    elif function_name in ("Rastrigin", "Schwefel", "BiSphere", "BiRastrigin"):
        ax.set_yscale("linear")
    else:
        ax.set_yscale("log")
    ax.legend()

    fig.show()

    save_path = Path(f"images/pso_{function_name.lower()}.png")
    if save_path.parent.exists() and not save_path.parent.is_dir():
        OSError("There is something in the way. We can't store our paintings.")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".svg"))
    fig.savefig(save_path.with_suffix(".pdf"))
    fig.savefig(save_path.with_stem(save_path.stem + "_T"), transparent=True)
    fig.savefig(save_path.with_stem(save_path.stem + "_T").with_suffix(".svg"), transparent=True)
    fig.savefig(save_path.with_stem(save_path.stem + "_T").with_suffix(".pdf"), transparent=True)
