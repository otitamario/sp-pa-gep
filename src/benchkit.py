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
    inner_hit_max: Optional[bool] = None
    ok: Optional[bool] = None
    constraint_violation: Optional[float] = None
    extras: Optional[Dict[str, Any]] = None


@dataclass
class RunSummary:
    method: str
    iters: int
    total_s: float
    avg_resolvent_s: float
    avg_residual_s: float
    final_step: Optional[float]
    final_error: Optional[float]
    final_residual: Optional[float]
    avg_inner_iters: Optional[float] = None
    max_inner_iters: Optional[int] = None
    pct_hit_max_inner: Optional[float] = None
    success_rate: Optional[float] = None
    final_constraint_violation: Optional[float] = None


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
    """
    Generic runner for iterative methods.

    step_fn(x, k) must return (x_next, info_dict).
    Preferred: info_dict["residual"] (float) computed inside step_fn.
    Fallback: if info_dict["u"] exists, residual = ||x_next - u||.
    Otherwise: if residual_fn provided, compute residual_fn(x_next).
    """
    logs: List[IterLog] = []
    x = np.array(x0, dtype=float).copy()

    total_resolvent_s = 0.0
    total_residual_s = 0.0

    with Timer() as total_timer:
        for k in range(max_iter):
            # --- main update / resolvent timing ---
            with Timer() as t_res:
                x_next, info = step_fn(x, k)
            resolvent_s = t_res.dt
            total_resolvent_s += resolvent_s

            step = float(np.linalg.norm(x_next - x))

            # --- residual (prefer cached values in info to avoid recomputation) ---
            residual = None
            residual_s = 0.0
            if residual_fn is not None:
                if isinstance(info, dict) and ("residual" in info):
                    residual = float(info["residual"])
                    residual_s = 0.0
                elif isinstance(info, dict) and ("u" in info):
                    residual = float(np.linalg.norm(x_next - info["u"]))
                    residual_s = 0.0
                else:
                    with Timer() as t_r:
                        residual = float(residual_fn(x_next))
                    residual_s = t_r.dt
                    total_residual_s += residual_s
            else:
                # If residual_fn is None, allow step_fn to supply residual anyway.
                if isinstance(info, dict) and ("residual" in info):
                    residual = float(info["residual"])
                elif isinstance(info, dict) and ("u" in info):
                    residual = float(np.linalg.norm(x_next - info["u"]))

            # --- error ---
            error = None
            if error_fn is not None:
                error = float(error_fn(x_next))

            # --- optional diagnostics extracted from info ---
            inner_iters = info.get("inner_iters", None) if isinstance(info, dict) else None
            inner_hit_max = info.get("inner_hit_max", None) if isinstance(info, dict) else None
            ok = info.get("ok", None) if isinstance(info, dict) else None
            constraint_violation = info.get("constraint_violation", None) if isinstance(info, dict) else None
            extras = info.get("extras", None) if isinstance(info, dict) else None

            logs.append(
                IterLog(
                    k=k,
                    step=step,
                    residual=residual,
                    error=error,
                    resolvent_s=resolvent_s,
                    residual_s=residual_s,
                    inner_iters=inner_iters,
                    inner_hit_max=inner_hit_max,
                    ok=ok,
                    constraint_violation=constraint_violation,
                    extras=extras,
                )
            )

            x = x_next
            if step <= stop_tol_step:
                break

    iters_done = len(logs)

    # Inner iteration summary
    inner_vals = [L.inner_iters for L in logs if L.inner_iters is not None]
    avg_inner = float(np.mean(inner_vals)) if inner_vals else None
    max_inner = int(np.max(inner_vals)) if inner_vals else None

    hit_vals = [L.inner_hit_max for L in logs if L.inner_hit_max is not None]
    pct_hit = float(np.mean(hit_vals)) if hit_vals else None  # mean of booleans

    ok_vals = [L.ok for L in logs if L.ok is not None]
    success_rate = float(np.mean(ok_vals)) if ok_vals else None

    cv_vals = [L.constraint_violation for L in logs if L.constraint_violation is not None]
    final_cv = cv_vals[-1] if cv_vals else None

    final_step = logs[-1].step if iters_done else None

    summary = RunSummary(
        method=method_name,
        iters=iters_done,
        total_s=total_timer.dt,
        avg_resolvent_s=(total_resolvent_s / iters_done) if iters_done else 0.0,
        avg_residual_s=(total_residual_s / iters_done) if iters_done else 0.0,
        final_step=final_step,
        final_error=logs[-1].error if iters_done else None,
        final_residual=logs[-1].residual if iters_done else None,
        avg_inner_iters=avg_inner,
        max_inner_iters=max_inner,
        pct_hit_max_inner=pct_hit,
        success_rate=success_rate,
        final_constraint_violation=final_cv,
    )
    return logs, summary


def _fmt_sci(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x == 0.0:
        return "0"
    return f"{x:.4e}"


def latex_table(summaries: List[RunSummary], caption: str, label: str) -> str:
    has_error = any(s.final_error is not None for s in summaries)
    has_inner = any(s.avg_inner_iters is not None for s in summaries)
    has_hit = any(s.pct_hit_max_inner is not None for s in summaries)
    has_ok = any(s.success_rate is not None for s in summaries)
    has_cv = any(s.final_constraint_violation is not None for s in summaries)

    cols = ["Method", "Iters", "Total (s)", "Avg resolvent (s)", "Avg residual (s)", "Final step"]
    if has_inner:
        cols.append("Avg inner iters")
        cols.append("Max inner iters")
    if has_hit:
        cols.append(r"\% hit max inner")
    if has_ok:
        cols.append("Success rate")
    if has_cv:
        cols.append("Final constr. viol.")
    if has_error:
        cols.append(r"Final $\|x_n-\bar x\|$")
    cols.append(r"Final $R(x_n)$")

    colspec = "l" + "r" * (len(cols) - 1)

    def row(s: RunSummary) -> str:
        fields = [
            s.method,
            f"{s.iters:d}",
            f"{s.total_s:.4f}",
            f"{s.avg_resolvent_s:.5f}",
            f"{s.avg_residual_s:.5f}",
            _fmt_sci(s.final_step),
        ]
        if has_inner:
            fields.append("-" if s.avg_inner_iters is None else f"{s.avg_inner_iters:.2f}")
            fields.append("-" if s.max_inner_iters is None else f"{s.max_inner_iters:d}")
        if has_hit:
            fields.append("-" if s.pct_hit_max_inner is None else f"{100.0*s.pct_hit_max_inner:.1f}")
        if has_ok:
            fields.append("-" if s.success_rate is None else f"{100.0*s.success_rate:.1f}")
        if has_cv:
            fields.append(_fmt_sci(s.final_constraint_violation))
        if has_error:
            fields.append(_fmt_sci(s.final_error))
        fields.append(_fmt_sci(s.final_residual))
        return " & ".join(fields) + r" \\"

    header = (
        "\\begin{table}[!ht]\n\\centering\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\hline\n"
        + " & ".join(cols) + "\\\\\n"
        "\\hline\n"
    )
    body = "\n".join(row(s) for s in summaries)
    footer = (
        "\n\\hline\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    return header + body + footer


def latex_tables_split(summaries: List[RunSummary],
                       caption_perf: str,
                       label_perf: str,
                       caption_acc: str,
                       label_acc: str) -> str:

    has_inner = any(s.avg_inner_iters is not None for s in summaries)
    has_hit = any(s.pct_hit_max_inner is not None for s in summaries)
    has_ok = any(s.success_rate is not None for s in summaries)
    has_cv = any(s.final_constraint_violation is not None for s in summaries)
    has_error = any(s.final_error is not None for s in summaries)

    # ---------- TABLE A: PERFORMANCE ----------
    cols_perf = ["Method", "It.", "Tot (s)", "Avg res (s)", "Step$_f$"]
    if has_inner:
        cols_perf += ["Avg in.", "Max in."]
    if has_hit:
        cols_perf.append(r"\% max")
    if has_ok:
        cols_perf.append("Succ. (\%)")

    colspec_perf = "l" + "r" * (len(cols_perf) - 1)

    def row_perf(s):
        fields = [
            s.method,
            f"{s.iters:d}",
            f"{s.total_s:.4f}",
            f"{s.avg_resolvent_s:.5f}",
            _fmt_sci(s.final_step),
        ]
        if has_inner:
            fields.append("-" if s.avg_inner_iters is None else f"{s.avg_inner_iters:.2f}")
            fields.append("-" if s.max_inner_iters is None else f"{s.max_inner_iters:d}")
        if has_hit:
            fields.append("-" if s.pct_hit_max_inner is None else f"{100.0*s.pct_hit_max_inner:.1f}")
        if has_ok:
            fields.append("-" if s.success_rate is None else f"{100.0*s.success_rate:.1f}")
        return " & ".join(fields) + r" \\"

    header_perf = (
        "\\begin{table}[!ht]\n\\centering\n"
        f"\\begin{{tabular}}{{{colspec_perf}}}\n\\hline\n"
        + " & ".join(cols_perf) + "\\\\\n"
        "\\hline\n"
    )
    body_perf = "\n".join(row_perf(s) for s in summaries)
    footer_perf = (
        "\n\\hline\n\\end{tabular}\n"
        f"\\caption{{{caption_perf}}}\n"
        f"\\label{{{label_perf}}}\n"
        "\\end{table}\n\n"
    )

    # ---------- TABLE B: ACCURACY ----------
    cols_acc = ["Method"]
    if has_cv:
        cols_acc.append("Constr. viol.")
    if has_error:
        cols_acc.append(r"$\|x_n-\bar x\|$")
    cols_acc.append(r"$R(x_n)$")

    colspec_acc = "l" + "r" * (len(cols_acc) - 1)

    def row_acc(s):
        fields = [s.method]
        if has_cv:
            fields.append(_fmt_sci(s.final_constraint_violation))
        if has_error:
            fields.append(_fmt_sci(s.final_error))
        fields.append(_fmt_sci(s.final_residual))
        return " & ".join(fields) + r" \\"

    header_acc = (
        "\\begin{table}[!ht]\n\\centering\n"
        f"\\begin{{tabular}}{{{colspec_acc}}}\n\\hline\n"
        + " & ".join(cols_acc) + "\\\\\n"
        "\\hline\n"
    )
    body_acc = "\n".join(row_acc(s) for s in summaries)
    footer_acc = (
        "\n\\hline\n\\end{tabular}\n"
        f"\\caption{{{caption_acc}}}\n"
        f"\\label{{{label_acc}}}\n"
        "\\end{table}\n"
    )

    return header_perf + body_perf + footer_perf + header_acc + body_acc + footer_acc

def make_standard_plots(
    *,
    logs_by_method: Dict[str, List[IterLog]],
    outdir: str,
    tag: str,
    plot_residual: bool = True,
    plot_error: bool = True,
    show_title: bool = False,
) -> Dict[str, str]:
    import numpy as np
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    outputs = {}

    # --- color assignment (same color per μ if present) ---
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

            if "μ=" in name:
                group = name.split("μ=")[1]
            else:
                group = name

            color = group_color[group]
            linestyle = "-" if "SPPA" in name else "--"
            plt.semilogy(np.arange(len(y)), np.array(y, float), linestyle=linestyle, color=color, linewidth=2.0, label=name)

        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        if show_title:
            plt.title(f"{tag}: {ylabel}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(frameon=False)
        plt.tight_layout()
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    outputs["step"] = plot_metric("step", r"$\|x_{n+1}-x_n\|$", f"{tag}_steps.png")
    if plot_residual:
        outputs["residual"] = plot_metric("residual", r"$R(x_n)$", f"{tag}_residual.png")
    if plot_error:
        outputs["error"] = plot_metric("error", r"$\|x_n-\bar{x}\|$", f"{tag}_error.png")

    return outputs
