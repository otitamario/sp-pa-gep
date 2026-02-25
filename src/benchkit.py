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
    _ensure_dir(outdir)
    out: Dict[str, str] = {}

    steps = {m: np.array([L.step for L in logs], float) for m, logs in logs_by_method.items()}
    p = os.path.join(outdir, f"{tag}_steps.png")
    _semilogy_multi(steps, r"$\|x_{n+1}-x_n\|$", f"{tag}: Step size", p)
    out["step"] = p

    if plot_residual:
        res = {}
        for m, logs in logs_by_method.items():
            vals = [L.residual for L in logs if L.residual is not None]
            if vals:
                res[m] = np.array(vals, float)
        if res:
            p = os.path.join(outdir, f"{tag}_residual.png")
            _semilogy_multi(res, r"$R(x_n)$", f"{tag}: Residual", p)
            out["residual"] = p

    if plot_error:
        err = {}
        for m, logs in logs_by_method.items():
            vals = [L.error for L in logs if L.error is not None]
            if vals:
                err[m] = np.array(vals, float)
        if err:
            p = os.path.join(outdir, f"{tag}_error.png")
            _semilogy_multi(err, r"$\|x_n-\bar{x}\|$", f"{tag}: Error", p)
            out["error"] = p

    return out


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
    logs_by_case: dict[str, list],  # case name -> list[IterLog]
    metric: str,                    # "step" | "residual" | "error"
    ylabel: str,
    title: str,
    outpath: str,
):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    plt.figure()
    for name, logs in logs_by_case.items():
        if metric == "step":
            y = [L.step for L in logs]
        elif metric == "residual":
            y = [L.residual for L in logs if L.residual is not None]
        elif metric == "error":
            y = [L.error for L in logs if L.error is not None]
        else:
            raise ValueError("metric must be one of: step, residual, error")

        if len(y) == 0:
            continue
        plt.semilogy(np.arange(len(y)), np.array(y, float), label=name)

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
