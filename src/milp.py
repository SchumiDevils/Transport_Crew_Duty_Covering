"""Solver-independent MILP container with HiGHS and Gurobi back-ends.

The models (src/models) only fill a `MILP` object (minimisation, row-wise
sparse matrix).  `solve()` then passes it to HiGHS (default, open source) or
Gurobi (needs a full licence for non-trivial sizes) with identical settings
and returns the same set of statistics for both.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

INF = math.inf


@dataclass
class MILP:
    cost: list[float] = field(default_factory=list)
    lb: list[float] = field(default_factory=list)
    ub: list[float] = field(default_factory=list)
    integer: list[bool] = field(default_factory=list)
    # row-wise sparse matrix (CSR)
    row_start: list[int] = field(default_factory=lambda: [0])
    row_index: list[int] = field(default_factory=list)
    row_value: list[float] = field(default_factory=list)
    row_lo: list[float] = field(default_factory=list)
    row_hi: list[float] = field(default_factory=list)

    # --------------------------------------------------------------- building
    def add_var(self, cost: float = 0.0, lb: float = 0.0, ub: float = 1.0, integer: bool = True) -> int:
        self.cost.append(cost)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integer.append(integer)
        return len(self.cost) - 1

    def add_row(self, idx, val, lo: float = -INF, hi: float = INF) -> None:
        self.row_index.extend(idx)
        self.row_value.extend(val)
        self.row_start.append(len(self.row_index))
        self.row_lo.append(lo)
        self.row_hi.append(hi)

    @property
    def num_vars(self) -> int:
        return len(self.cost)

    @property
    def num_rows(self) -> int:
        return len(self.row_lo)

    @property
    def nnz(self) -> int:
        return len(self.row_index)

    @property
    def num_int(self) -> int:
        return sum(self.integer)


@dataclass
class SolveResult:
    status: str            # optimal | time_limit | infeasible | error...
    obj: float             # best integer objective (nan if none)
    bound: float           # best dual bound
    gap: float             # (obj - bound) / obj
    root_bound: float      # dual bound at the end of the root node (after cuts)
    nodes: int
    runtime: float         # wall-clock seconds of the MIP solve
    x: np.ndarray | None   # best solution


def _gap(obj: float, bound: float) -> float:
    if not math.isfinite(obj) or not math.isfinite(bound):
        return math.nan
    return max(0.0, (obj - bound) / max(abs(obj), 1e-9))


# =============================================================================
# HiGHS
# =============================================================================
def _highs_model(m: MILP, relax: bool):
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_ = m.num_vars
    lp.num_row_ = m.num_rows
    lp.col_cost_ = np.asarray(m.cost, dtype=float)
    lp.col_lower_ = np.asarray(m.lb, dtype=float)
    lp.col_upper_ = np.asarray(m.ub, dtype=float)
    lp.row_lower_ = np.asarray(m.row_lo, dtype=float)
    lp.row_upper_ = np.asarray(m.row_hi, dtype=float)
    a = lp.a_matrix_
    a.format_ = highspy.MatrixFormat.kRowwise
    a.num_col_ = m.num_vars
    a.num_row_ = m.num_rows
    a.start_ = np.asarray(m.row_start, dtype=np.int32)
    a.index_ = np.asarray(m.row_index, dtype=np.int32)
    a.value_ = np.asarray(m.row_value, dtype=float)
    if not relax:
        t_int, t_cont = highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous
        lp.integrality_ = [t_int if b else t_cont for b in m.integer]
    return lp


def solve_highs(m: MILP, time_limit: float, threads: int = 1, relax: bool = False,
                log_file: str | None = None, seed: int = 0) -> SolveResult:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", log_file is not None)
    h.setOptionValue("log_to_console", False)
    if log_file:
        h.setOptionValue("log_file", log_file)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("random_seed", int(seed))
    h.passModel(_highs_model(m, relax))

    root = {"bound": -INF}

    def on_interrupt(e):  # record the dual bound while still at the root node
        d = e.data_out
        if d.mip_node_count == 0 and math.isfinite(d.mip_dual_bound):
            root["bound"] = d.mip_dual_bound

    if not relax:
        h.cbMipInterrupt.subscribe(on_interrupt)

    t0 = time.perf_counter()
    h.run()
    runtime = time.perf_counter() - t0

    ms = h.getModelStatus()
    info = h.getInfo()
    S = highspy.HighsModelStatus
    status = {S.kOptimal: "optimal", S.kTimeLimit: "time_limit", S.kInfeasible: "infeasible",
              S.kMemoryLimit: "memory_limit"}.get(ms, str(ms))
    has_sol = info.primal_solution_status == 2  # kSolutionStatusFeasible
    x = np.asarray(h.getSolution().col_value) if has_sol else None
    obj = info.objective_function_value if has_sol else math.nan
    if relax:
        bound = obj if status == "optimal" else math.nan
        return SolveResult(status, obj, bound, 0.0, bound, 0, runtime, x)
    bound = info.mip_dual_bound
    nodes = int(info.mip_node_count)
    rb = root["bound"]
    if not math.isfinite(rb) and nodes <= 1:
        rb = bound
    return SolveResult(status, obj, bound, _gap(obj, bound), rb, nodes, runtime, x)


# =============================================================================
# Gurobi
# =============================================================================
def solve_gurobi(m: MILP, time_limit: float, threads: int = 1, relax: bool = False,
                 log_file: str | None = None, seed: int = 0) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    g = gp.Model(env=env)
    g.Params.TimeLimit = time_limit
    g.Params.Threads = threads
    g.Params.Seed = seed
    if log_file:
        g.Params.LogFile = log_file
        g.Params.LogToConsole = 0
    vt = [GRB.CONTINUOUS if relax or not b else GRB.BINARY if (lo == 0 and up == 1) else GRB.INTEGER
          for b, lo, up in zip(m.integer, m.lb, m.ub)]
    x = g.addVars(m.num_vars, lb=m.lb, ub=m.ub, obj=m.cost, vtype=vt)
    for r in range(m.num_rows):
        a, b = m.row_start[r], m.row_start[r + 1]
        expr = gp.LinExpr(m.row_value[a:b], [x[i] for i in m.row_index[a:b]])
        lo, hi = m.row_lo[r], m.row_hi[r]
        if lo == hi:
            g.addLConstr(expr, GRB.EQUAL, lo)
        elif lo == -INF:
            g.addLConstr(expr, GRB.LESS_EQUAL, hi)
        elif hi == INF:
            g.addLConstr(expr, GRB.GREATER_EQUAL, lo)
        else:
            g.addRange(expr, lo, hi)
    g.ModelSense = GRB.MINIMIZE

    root = {"bound": -INF}

    def cb(model, where):
        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            root["bound"] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

    t0 = time.perf_counter()
    g.optimize(None if relax else cb)
    runtime = time.perf_counter() - t0
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
              GRB.INFEASIBLE: "infeasible"}.get(g.Status, str(g.Status))
    has_sol = g.SolCount > 0
    obj = g.ObjVal if has_sol else math.nan
    xs = np.array([x[i].X for i in range(m.num_vars)]) if has_sol else None
    if relax:
        return SolveResult(status, obj, obj, 0.0, obj, 0, runtime, xs)
    bound = g.ObjBound
    rb = root["bound"] if math.isfinite(root["bound"]) else bound
    return SolveResult(status, obj, bound, _gap(obj, bound), rb, int(g.NodeCount), runtime, xs)


def solve(m: MILP, solver: str = "highs", **kw) -> SolveResult:
    return {"highs": solve_highs, "gurobi": solve_gurobi}[solver](m, **kw)
