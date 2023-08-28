import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

functions = ("Sphere", "Rosenbrock", "Step", "Quartic", "Griewank", "Rastrigin", "Schwefel", "BiSphere", "BiRastrigin")
path = Path("./results3/")


def insert_data(d_array, idx, pt):
    if not p.is_dir() or len([f for f in p.iterdir()]) == 0:
        return
    for file in pt.iterdir():
        if not file.suffix == ".pkl":
            continue
        with open(file, "rb") as f:
            tmp = pickle.load(f, fix_imports=True)
            d_array[idx].append([min(tmp, key=lambda v: v.loss).loss, (max(tmp, key=lambda v: v.rank).rank + 1) / 64])


def refine_value(raw_value) -> int:
    for x in range(5):
        if raw_value < 2 ** x:
            return 2 ** (x - 1)
    else:
        return 16


if __name__ == "__main__":
    for function_name in functions:
        data = []
        pso_names = ("VelocityClamping", "Constriction", "Basic", "Canonical")
        marker_list = ("o", "s", "D", "^", "P", "X")  # ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]
        # np.random.shuffle(marker_list)
        for i in range(5):
            data.append([])
            if i == 4:
                d = f"bm_P_{function_name.lower()}_?"
            else:
                d = f"bm_{i}_{function_name.lower()}_?"
            for p in path.glob(d):
                insert_data(data, i, p)
            data[i] = np.array(sorted(data[i], key=lambda v: v[1])).T
        data.append([])
        for p in path.glob(f"bm_H_{function_name.lower()}_?"):
            if not p.is_dir():
                continue
            file = p / Path("result_0.pkl")
            with open(file, "rb") as f:
                tmp = pickle.load(f, fix_imports=True)
                data[-1].append([min(tmp[0]["losses"]), 2000 // len(tmp[0])])
                if data[-1][-1][1] not in (1, 2, 4, 8, 16):
                    data[-1][-1][1] = refine_value(data[-1][-1][1])
        data[5] = np.array(sorted(data[5], key=lambda v: v[1])).T

        fig: Figure
        ax: Axes

        fig, ax = plt.subplots()
        # fig.subplots_adjust(hspace=0)

        ax.set_title(f"PSO@Propulate on {function_name} function")
        ax.set_xlabel("Nodes")

        for i in range(4):
            # if function_name == "Schwefel" and i < len(pso_names) and pso_names[i] == "VelocityClamping":
            #     continue
            ax.plot(data[i][1], data[i][0], label=pso_names[i], marker=marker_list[i], ls="dotted", lw=2)
        ax.plot(data[4][1], data[4][0], label="Vanilla Propulate", marker=marker_list[4], lw=1, ms=8)
        ax.plot(data[5][1], data[5][0], label="Hyppopy", marker=marker_list[5], lw=1, ms=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
        ax.grid(True)
        ax.set_ylabel("Loss")
        if function_name == "Rosenbrock":
            ax.set_yscale("symlog", linthresh=1e-36)
            ax.set_yticks([0, 1e-36, 1e-30, 1e-24, 1e-18, 1e-12, 1e-6, 1])
            ax.set_ylim(-5e-36, 1)
        elif function_name == "Step":
            ax.set_yscale("symlog")
            ax.set_ylim(-1e5, -5)
        elif function_name == "Schwefel":
            ax.set_yscale("symlog")
            ax.set_ylim(-50000, 5000)
        elif function_name in ("Schwefel", "Rastrigin", "BiSphere", "BiRastrigin"):
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
