# benchkit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List, Tuple
import time
import os
import numpy as np
import matplotlib.pyplot as plt


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0


@dataclass
class IterLog:
    k: int
    step: float
    residual: Optional[float]
    error: Optional[float]
    resolvent_s: float
    residual_s: float
    inner_iters: Optional[int] = None


@dataclass
class RunSummary:
    method: str
    iters: int
    total_s: float
    avg_resolvent_s: float
    avg_residual_s: float
    final_error: Optional[float]
    final_residual: Optional[float]


def run(
    *,
    method_name: str,
    x0: np.ndarray,
    max_iter: int,
    stop_tol_step: float,
    step_fn: Callable[[np.ndarray, int], Tuple[np.ndarray, Dict[str, Any]]],
    residual_fn: Optional[Callable[[np.ndarray], float]] = None,
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Tuple[List[IterLog], RunSummary]:

    logs: List[IterLog] = []
    x = np.array(x0, dtype=float).copy()
    total_resolvent_s = 0.0
    total_residual_s = 0.0

    with Timer() as total_timer:
        for k in range(max_iter):
            with Timer() as t_res:
                x_next, info = step_fn(x, k)
            resolvent_s = t_res.dt
            total_resolvent_s += resolvent_s

            step = float(np.linalg.norm(x_next - x))

            residual = None
            residual_s = 0.0
            if residual_fn is not None:
                with Timer() as t_r:
                    residual = float(residual_fn(x_next))
                residual_s = t_r.dt
                total_residual_s += residual_s

            error = None
            if error_fn is not None:
                error = float(error_fn(x_next))

            logs.append(
                IterLog(
                    k=k, step=step, residual=residual, error=error,
                    resolvent_s=resolvent_s, residual_s=residual_s,
                    inner_iters=info.get("inner_iters", None),
                )
            )

            x = x_next
            if step <= stop_tol_step:
                break

    iters_done = len(logs)
    summary = RunSummary(
        method=method_name,
        iters=iters_done,
        total_s=total_timer.dt,
        avg_resolvent_s=(total_resolvent_s / iters_done) if iters_done else 0.0,
        avg_residual_s=(total_residual_s / iters_done) if iters_done else 0.0,
        final_error=logs[-1].error if iters_done else None,
        final_residual=logs[-1].residual if iters_done else None,
    )
    return logs, summary


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _semilogy_multi(curves: Dict[str, np.ndarray], ylabel: str, title: str, outpath: str) -> None:
    plt.figure()
    for name, y in curves.items():
        if y is None or len(y) == 0:
            continue
        plt.semilogy(np.arange(len(y)), y, label=name)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_standard_plots(
    *,
    logs_by_method: Dict[str, List[IterLog]],
    outdir: str,
    tag: str,
    plot_residual: bool = True,
    plot_error: bool = True,
) -> Dict[str, str]:

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    outputs = {}

    # --- color assignment (same color per μ if present) ---
    # Extract μ if present in method name
    unique_groups = set()
    for name in logs_by_method.keys():
        if "μ=" in name:
            unique_groups.add(name.split("μ=")[1])
        else:
            unique_groups.add(name)

    unique_groups = sorted(unique_groups)
    cmap = plt.get_cmap("tab10")
    group_color = {g: cmap(i) for i, g in enumerate(unique_groups)}

    def plot_metric(metric_key, ylabel, filename):
        plt.figure()

        for name, logs in logs_by_method.items():

            if metric_key == "step":
                y = [L.step for L in logs]
            elif metric_key == "residual":
                y = [L.residual for L in logs if L.residual is not None]
            elif metric_key == "error":
                y = [L.error for L in logs if L.error is not None]
            else:
                continue

            if len(y) == 0:
                continue

            # determine grouping (μ or method name)
            if "μ=" in name:
                group = name.split("μ=")[1]
            else:
                group = name

            color = group_color[group]
            linestyle = "-" if "SPPA" in name else "--"

            plt.semilogy(
                np.arange(len(y)),
                np.array(y, float),
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
                label=name,
            )

        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(f"{tag}: {ylabel}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(frameon=False)
        plt.tight_layout()

        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # --- Step size ---
    outputs["step"] = plot_metric(
        "step",
        r"$\|x_{n+1}-x_n\|$",
        f"{tag}_steps.png"
    )

    # --- Residual ---
    if plot_residual:
        outputs["residual"] = plot_metric(
            "residual",
            r"$R(x_n)$",
            f"{tag}_residual.png"
        )

    # --- Error ---
    if plot_error:
        outputs["error"] = plot_metric(
            "error",
            r"$\|x_n-\bar{x}\|$",
            f"{tag}_error.png"
        )

    return outputs

def _fmt_sci(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x == 0.0:
        return "0"
    return f"{x:.4e}"


def latex_row(s: RunSummary) -> str:
    return (
        f"{s.method} & {s.iters:d} & {s.total_s:.4f} & "
        f"{s.avg_resolvent_s:.5f} & {s.avg_residual_s:.5f} & "
        f"{_fmt_sci(s.final_error)} & {_fmt_sci(s.final_residual)} \\\\"
    )


def latex_table(summaries: List[RunSummary], caption: str, label: str) -> str:
    header = (
        "\\begin{table}[!ht]\n\\centering\n"
        "\\begin{tabular}{lrrrrrr}\n\\hline\n"
        "Method & Iters & Total (s) & Avg resolvent (s) & Avg residual (s) & "
        "Final $\\|x_n-\\bar x\\|$ & Final $R(x_n)$\\\\\n"
        "\\hline\n"
    )
    body = "\n".join(latex_row(s) for s in summaries)
    footer = (
        "\n\\hline\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return header + body + footer


def plot_metric_across_cases(
    *,
    logs_by_case: dict,
    metric: str,
    ylabel: str,
    title: str,
    outpath: str,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    plt.figure()

    # consistent color per μ
    unique_mu = sorted({
        name.split("μ=")[1] for name in logs_by_case.keys()
    })

    cmap = plt.get_cmap("tab10")
    mu_color = {mu: cmap(i) for i, mu in enumerate(unique_mu)}

    for name, logs in logs_by_case.items():

        if metric == "step":
            y = [L.step for L in logs]
        elif metric == "residual":
            y = [L.residual for L in logs if L.residual is not None]
        elif metric == "error":
            y = [L.error for L in logs if L.error is not None]
        else:
            raise ValueError("metric must be 'step', 'residual' or 'error'")

        if len(y) == 0:
            continue

        mu = name.split("μ=")[1]
        color = mu_color[mu]

        linestyle = "-" if "SPPA" in name else "--"

        plt.semilogy(
            np.arange(len(y)),
            np.array(y, float),
            linestyle=linestyle,
            color=color,
            linewidth=2.0,
            label=name
        )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
