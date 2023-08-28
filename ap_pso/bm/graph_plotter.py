import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

functions = ("Sphere", "Rosenbrock", "Step", "Quartic", "Rastrigin", "Griewank", "Schwefel", "BiSphere", "BiRastrigin")
pso_names = ("VelocityClamping", "Constriction", "Basic", "Canonical")
other_stuff = ("Vanilla Propulate", "Hyppopy")

time_path = Path("./slurm3/")
path = Path("./results3/")


def insert_data(d_array, idx: int, pt: Path):
    """
    This function adds the data given via `pt` into the data array given by `d_array` at position `idx`.
    """
    if not p.is_dir() or len([f for f in p.iterdir()]) == 0:
        return
    for fil in pt.iterdir():
        if not fil.suffix == ".pkl":
            continue
        with open(fil, "rb") as f:
            tm = pickle.load(f, fix_imports=True)
            d_array[idx].append([min(tm, key=lambda v: v.loss).loss, (max(tm, key=lambda v: v.rank).rank + 1) / 64])


def refine_value(raw_value) -> int:
    """
    This function ensures that values that are larger than they should be, are corrected to the correct number of cores.
    """
    for x in range(5):
        if raw_value < 2 ** x:
            return 2 ** (x - 1)
    else:
        return 16


def calc_time(iterator) -> float:
    """
    This function takes an iterator on a certain string array and calculates out of this a time span in seconds.
    """
    try:
        start = int(next(iterator).strip("\n|: Ceirmnrtu"))
    except ValueError:
        return np.nan
    try:
        end = int(next(iterator).strip("\n|: Ceirmnrtu"))
    except ValueError:
        return np.nan
    return (end - start) / 1e9


if __name__ == "__main__":
    raw_time_data: list[str] = []
    time_data: dict[str, dict[str, list[float]]] = {}

    for function_name in functions:
        time_data[function_name] = {}
        for program in other_stuff + pso_names:
            time_data[function_name][program] = []

    for file in time_path.iterdir():
        with open(file) as f:
            raw_time_data.append(f.read())

    for x in raw_time_data:
        scatter = [st for st in x.split("#-----------------------------------#") if "Current time" in st]
        itx = iter(scatter)
        for program in other_stuff:
            for function_name in functions:
                time_data[function_name][program].append(calc_time(itx))
        for function_name in functions:
            for program in pso_names:
                time_data[function_name][program].append(calc_time(itx))

    for function_name in functions:
        data = []
        marker_list = ("o", "s", "D", "^", "P", "X")  # ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D"]

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
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
        ax.grid(True)
        ax.set_ylabel("Loss")

        ax_t = ax.twinx()
        ax_t.set_ylabel("Time [s]")
        ax_t.set_yscale("log")

        everything = pso_names + other_stuff
        for i, name in enumerate(everything):
            if i < 4:
                ms = 6
            else:
                ms = 7
            ax.plot(data[i][1], data[i][0], label=name, marker=marker_list[i], ls="dashed", lw=2, ms=ms)
            ax_t.plot(data[i][1], time_data[function_name][name], marker=marker_list[i], ls="dotted", ms=ms)

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
