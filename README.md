# Project n. 2 — Transport Crew Duty Covering (Project Plan & Step-by-Step Guide)

**Course:** Model-driven Decision Methods (Università di Pisa)  
**Track:** Normal Project  
**Status:** Phase 0 done (except Python install + names) → next: Phase 1

> **Source documents.** The binding document for this project is *“Model-driven Decision Methods: Project”* (`projects-proposals.pdf`, steps 1–6 + report contents). The other file, `projects-instructions.pdf`, is the guide of a **different course** (*Optimization for Data Science, 784AA*). Its general workflow advice (partial reports, plagiarism, email-only communication) is a good fit, but its convergence-theory requirements (§3.3) do **not** apply here. Where the two disagree, follow the MDDM document and ask the lecturer.

---

## 1. Problem Statement (Verbatim)

> **Project n. 2**  
> A transport company has to cover a set of driving duties with the available personnel. Each employee can perform different subsets of the duties; the subsets are implicitly specified under the form of a (given) “compatibility graph”. That is, for each duty its starting time and the ending time are given; furthermore, for any pair of duties it is known the time that it takes to move from the place where the first one ends to the place where the other one starts, so that it is known whether or not it is possible for the same employee to perform both in sequence. Union regulations set for each employee a maximum working time (sum of the driving time) and a maximum total time (difference between the start of his first duty and the end of its last one). Employees can be both half-time or full-time; the former have maximum working time and total time half than the latter, but cost 60% as much. The problem is to assign a feasible set of duties to an appropriate number of employees (of which there is an unlimited supply, of either type) in order to have each duty performed by exactly one employee, at minimum total cost.

### Quick Problem Breakdown

| Component | Details |
| :--- | :--- |
| **Duties** | Set $D = \{1, \dots, n\}$; each duty $i$ has start time $s_i$, end time $e_i$, and driving duration $d_i = e_i - s_i$. |
| **Compatibility Graph** | Directed Acyclic Graph (DAG) $G = (D, A)$ where arc $(i, j) \in A \iff e_i + t_{ij} \le s_j$, with $t_{ij}$ being the transfer time from the end location of $i$ to the start location of $j$. |
| **Employee Types** | Unlimited supply of **Full-Time (FT)** and **Half-Time (HT)** employees. |
| **Union Regulations** | **FT:** max working time (sum of $d_i$) $\le W$, max total spread ($e_{\text{last}} - s_{\text{first}}$) $\le T$, cost $= C$ (e.g., $1.0$).<br>**HT:** max working time $\le W/2$, max total spread $\le T/2$, cost $= 0.6C$ (e.g., $0.6$). |
| **Goal** | Partition $D$ into feasible employee shifts so every duty is covered by **exactly one** employee at **minimum total cost**. |

---

## 2. Recommended Repository Layout

> **Note:** Keep any Git repository **private** (the lecturer prefers it; not strictly mandatory for normal projects). The MDDM document asks for **one** `.zip`/`.tgz` holding the final PDF report *and* all support files. The “code only after report approval” rule comes from the ODS document, so confirm the order with the lecturer.

```text
project/
├── README.md                 # This project plan & execution guide
├── data/
│   ├── instances/            # Generated (or downloaded) benchmark instances
│   └── generator.py          # Scalable parameterized instance generator
├── src/
│   ├── models/               # Baseline MILP(s) & improved formulations / CG
│   ├── heuristics/           # Constructive/greedy bounds & warm-starts
│   ├── validator.py          # Independent feasibility & cost checker
│   └── run_experiments.py    # Automated benchmark runner
├── results/
│   ├── raw/                  # Solver logs & CSV outputs
│   └── plots/                # Generated figures & LaTeX tables
└── report/
    ├── main.tex              # LaTeX report (incremental across phases)
    └── main.pdf              # Compiled PDF report for email submissions
```

---

## 3. Step-by-Step Execution Plan

The course recommends delivering the report in **three incremental phases** via email so conceptual mistakes are caught early before heavy coding or experiments.

---

### Phase 0: Environment & Project Setup

- [x] **Choose your tech stack** (Normal project gives full freedom). **Decision:**
  - **Language:** Python 3.12.
  - **Modeling + solver:** `gurobipy` + Gurobi (main solver; free *named academic license* from gurobi.com/academia, activated on the university network or VPN). Use `highspy` (HiGHS, open source) as a second solver to cross-check results.
  - Pinned in [requirements.txt](requirements.txt).
- [ ] **Install the environment** (still to do: no Python on this machine yet):
  ```powershell
  winget install Python.Python.3.12
  py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  grbgetkey <your-academic-key>   # after requesting the license on gurobi.com
  ```
- [x] **Initialize workspace & LaTeX report template**:
  - [x] Folder layout created (`data/`, `src/models`, `src/heuristics`, `results/raw`, `results/plots`, `report/`).
  - [x] [report/main.tex](report/main.tex): title page with project number + verbatim statement, six section stubs that follow the MDDM report contents. It builds with `latexmk -pdf main.tex`.
  - [ ] Fill in **group number, names, matricola, e-mails** (marked `TODO` in red in the PDF).
  - [x] Local Git repo initialised (`main` branch, `.gitignore`). If you add a remote (GitHub/GitLab), make it **private**.

---

### Phase 1: Mathematical Modeling & Instance Generation (Official Steps 1–2)

#### Step 1 — Mathematical Programming Formulation(s)

> **Step 1 status: written** in [report/sections/model.tex](report/sections/model.tex) (§1.1–1.5 of the PDF). Contents: assumptions A1–A5; models **(AF)** arc-flow, **(CF)** conflict, **(CF-M)** big-M variant, **(SP)** set partitioning; Propositions 1–6 (transitivity, correctness of AF/CF, LP-bound comparison, strong NP-hardness). Questions to put in the Milestone 1 e-mail: A1 (deadhead counted as working time?) and the order of report vs. code submission.

- [x] **Clarify domain details & assumptions** (document clearly in the report and verify with the lecturer if needed):
  - Confirm that transfer time $t_{ij}$ affects feasibility ($e_i + t_{ij} \le s_j$) and total spread ($e_{\text{last}} - s_{\text{first}}$), while working time counts strictly the sum of driving times $\sum d_i$.
  - Confirm feasibility assumptions for single duties (e.g., every duty satisfies $d_i \le W$ and $e_i - s_i \le T$; duties with $d_i > W/2$ can only be covered by FT employees).
  - Note whether transfer times satisfy the triangle inequality (which makes $G$ transitive) or are general (requiring explicit path/flow sequencing).
  - **Open question (ask the lecturer):** does “working time (sum of the driving time)” include the *deadhead* driving $t_{ij}$ between duties, or only $\sum d_i$? The plan assumes only $\sum d_i$.
  - **Instance sanity:** working time ≤ spread always holds, so the working-time limit only matters if $W < T$. The generator must enforce this, or that constraint never binds.
  - Cost is a **fixed cost per employee used** (1 for FT, 0.6 for HT), independent of the hours worked.
- [x] **Derive the Baseline Compact MILP Formulation(s)**:
  - **Option A — Employee-Indexed Arc-Flow Formulation** (works for any compatibility DAG $G$):
    - Define a pool of candidate FT and HT employees (sized using an upper bound, e.g., $n$ or a heuristic bound).
    - Binary variables: employee activation $y_k$, duty assignment $x_{ik}$, and sequence arcs $z_{ijk}$ on a source/sink augmented DAG.
    - Constraints: exact duty coverage, flow conservation per active employee, working time limit $\sum_i d_i x_{ik} \le W_k y_k$, and total spread limit $e_{\text{last}} - s_{\text{first}} \le T_k$.
  - **Option B — Employee-Indexed Assignment / Conflict Formulation** (when $G$ is transitive):
    - Avoids $O(|A| \cdot |K|)$ arc variables by forbidding incompatible pairs $(i, j)$ on the same employee: $x_{ik} + x_{jk} \le y_k$.
    - **Key insight for spread:** $\max_{j \in S} e_j - \min_{i \in S} s_i \le T_k \iff e_j - s_i \le T_k\;\forall i, j \in S$. You can compare a naive Big-$M$ spread formulation against a tighter pairwise-incompatibility formulation.
- [x] **Outline an Extended Formulation (for Phase 3)**:
  - Formulate the **Set Partitioning (Master) Problem** over all feasible FT and HT shifts (paths in $G$), setting up the theoretical foundation for Column Generation / Branch-and-Price later.
- [x] **Analyze model size & theoretical properties**:
  - Count variables and constraints as functions of $|D|$, $|A|$, and employee pool size $|K|$.
  - Note structural properties (DAG acyclicity, employee symmetry, LP relaxation strength, NP-hardness). *Correction:* a naive reduction from Bin Packing/Partition does **not** work, because a mix of HT and FT can undercut it. The report uses a tight **3-Partition** reduction instead, which gives *strong* NP-hardness.

#### Step 2 — Benchmark Instances & Scalable Generator

- [ ] **Search for existing instances**:
  - Check public transport / crew scheduling benchmarks (e.g., Bus Driver Scheduling / Vehicle Scheduling instances) to see if any can be adapted.
- [ ] **Design and implement a scalable generator (`data/generator.py`)**:
  - Make it deterministic via configurable random seeds and save instances in a clean format (JSON or plain text).
  - Include parameters that **realistically control problem difficulty**:
    1. **Instance size ($n = |D|$):** e.g., Tiny ($10$–$15$ for verification), Small ($20$–$40$), Medium ($50$–$100$), Large ($150$–$300+$).
    2. **Time horizon & duty distribution:** uniform across the day vs. realistic peak hours (morning/evening rush waves, which create high concurrency and drive employee count).
    3. **Duty durations ($d_i$):** short vs. long duties, including mixes that make FT vs. HT trade-offs non-trivial (e.g., awkward remainders where HT is cheaper than a lightly loaded FT shift).
    4. **Spatial / Transfer matrix ($t_{ij}$):** e.g., $L$ depots/terminals in 2D space with asymmetric or Euclidean travel times, controlling **compatibility graph density**.
    5. **Union regulation tightness ($W, T$):** ratio $W / T$ (tight driving limit vs. tight spread limit) so both constraints actively bind in different instance classes.
- [ ] **Generate a structured benchmark suite** spanning sizes and difficulty regimes, and document its summary statistics (graph density, average duty length, theoretical minimum FT/HT bounds).

> **Milestone 1 (Email Lecturer):** Write Sections 1–2 of the PDF report (problem description, mathematical formulation(s), instance generator design & dataset summary) and send it to the lecturer for early review before running large-scale experiments.

---

### Phase 2: Baseline Implementation & Computational Analysis (Official Steps 3–4)

#### Step 3 — Initial Implementation & Verification

- [ ] **Implement an independent solution validator (`src/validator.py`)**:
  - Given an instance and a solver output, verify from scratch:
    - Every duty is assigned to **exactly one** employee.
    - Each employee's duties form a valid directed path in the compatibility graph $G$.
    - Sum of driving times $\le W$ (FT) or $\le W/2$ (HT).
    - Spread $(e_{\text{last}} - s_{\text{first}}) \le T$ (FT) or $\le T/2$ (HT).
    - Recomputed total cost matches the solver's reported objective.
- [ ] **Implement the baseline MILP model(s)** (`src/models/baseline.py`):
  - Keep solver parameters at **default settings** for a clean baseline.
  - Enforce a consistent wall-clock time limit (e.g., 300s or 600s) and fixed thread count across all runs.
- [ ] **Sanity-test on tiny instances** ($n = 5$–$15$):
  - Manually verify optimal solutions on hand-crafted edge cases (e.g., all HT optimal, all FT optimal, mixed FT+HT optimal, tight spread vs. tight working time).
- [ ] **Build an automated experiment runner (`src/run_experiments.py`)**:
  - Run across the full benchmark suite and log per-instance metrics to CSV: `instance`, `n`, `density`, `status`, `best_obj`, `best_bound`, `mip_gap`, `root_lp_bound`, `runtime_sec`, `bb_nodes`, `num_ft`, `num_ht`, `valid`.

#### Step 4 — Initial Performance Analysis & Bottleneck Diagnosis

- [ ] **Summarize baseline performance** with clean tables and readable plots (use log scales for runtimes/gaps):
  - Runtime and MIP gap vs. instance size $n$.
  - Effect of compatibility density, peakiness, and $W/T$ tightness on hardness.
  - Maximum instance size reliably solved to optimality on your machine.
- [ ] **Identify the primary bottlenecks** (form concrete hypotheses for Step 5):
  - **Symmetry:** Are many Branch-and-Bound nodes wasted exploring permutations of identical FT/HT employees?
  - **Weak LP relaxation:** Is there a large integrality gap at the root node (e.g., fractional $y_k$ or loose Big-$M$ spread constraints)?
  - **Model blow-up:** Does the number of variables/constraints scale too fast with $n$ and $|A|$?
  - **Poor primal bounds:** Does the solver struggle to find good feasible integer solutions early?

> **Milestone 2 (Email Lecturer):** Add Sections 3–4 to the PDF report (implementation choices, baseline experimental setup, tables/plots, bottleneck analysis) and email the updated report for feedback.

---

### Phase 3: Performance Improvements & Comparative Evaluation (Official Steps 5–6)

#### Step 5 — Propose & Implement Improvements

Select **at least one** (ideally 2–3 complementary or progressively stronger) improvement strategies directly targeting the bottlenecks found in Step 4:

- [ ] **Track A — Model Strengthening & Preprocessing (Compact MILP)**
  - **Graph & domain preprocessing:** Remove HT-infeasible duties ($d_i > W/2$) and infeasible arcs/pairs where $e_j - s_i > T_k$ or $d_i + d_j > W_k$; compute a tight upper bound on $|K_{\text{FT}}|$ and $|K_{\text{HT}}|$ via a fast greedy heuristic.
  - **Symmetry breaking:** Enforce non-increasing employee activation ($y_1 \ge y_2 \ge \dots$) and lexicographic / minimum-duty-index ordering (e.g., employee $k$ can only be assigned duty $i$ if $i \ge k$, or ordered by earliest start time).
  - **Valid inequalities:** Add clique inequalities on the incompatibility graph (duties overlapping in time cannot share an employee), disaggregated constraints, or replace Big-$M$ spread constraints with pairwise conflict cuts.
- [ ] **Track B — Reformulation via Column Generation / Branch-and-Price (High-Impact Approach)**
  - **Restricted Master Problem (RMP):** Linear relaxation of the Set Partitioning / Set Covering formulation initialized with heuristic shifts.
  - **Pricing Subproblems (one for FT, one for HT):** Find a feasible shift (path in DAG $G$) with negative reduced cost $c_{\text{type}} - \sum_{i \in p} \pi_i < 0$ subject to working time and spread limits — a **Resource-Constrained Shortest Path Problem (RCSPP)** solvable via dynamic programming / labeling along a topological ordering of $G$.
  - **Integer solutions:** Implement either **Price-and-Branch** (solve root LP via CG, then solve RMP as an integer program — simple and often near-optimal) or full **Branch-and-Price** (with Ryan-Foster branching on duty pairs).
- [ ] **Track C — Heuristic Warm-Starts & Solver Parameter Tuning**
  - Implement a fast constructive + local-search heuristic (e.g., greedy path packing / shift merging/splitting) to feed a strong initial MIP start to the solver.
  - Systematically test targeted solver parameters (e.g., symmetry level, cut aggressiveness, MIP focus, branching priorities on $y_k$ variables) and measure their isolated impact.

#### Step 6 — Comparative Computational Evaluation

- [ ] **Run controlled apples-to-apples experiments**:
  - Evaluate the baseline and each proposed improvement on the **exact same benchmark instances**, hardware, thread limit, and time limit.
  - Verify every solution with `validator.py` and confirm matching optimal objectives on all instances solved to optimality by both methods.
- [ ] **Analyze & visualize the impact**:
  - Report speedup factors, reduction in MIP gap, root LP gap improvement, B&B node reduction, and increase in maximum solvable instance size.
  - Use concise, readable visuals: performance profiles (Dolan–Moré), log-scale runtime vs. $n$ curves, root gap comparison bar charts, and compact summary tables (avoid multi-page raw number dumps).
  - Clearly discuss **trade-offs** (e.g., overhead of CG on small instances vs. massive gains on large instances, or which improvement contributes most).

> **Milestone 3 (Email Lecturer):** Complete Sections 5–6 of the PDF report (description of improvements + comparative evaluation & conclusions) and submit the full report for final approval.

---

## 4. Final Deliverables & Oral Exam Checklist

### A. Final Report (PDF, sent via email first)

Verify that the PDF report includes all mandatory items:
- [ ] **Title page:** Group number, member names/matricola/emails, project number, and verbatim problem statement.
- [ ] **Section 1:** Description of the optimization problem and mathematical model(s) + notation & theoretical discussion.
- [ ] **Section 2:** Description of the instances and scalable generator design.
- [ ] **Section 3:** Description of the initial implementation (tools, language, solver) with justification of choices.
- [ ] **Section 4:** First computational experiments with clean tables/graphs and bottleneck analysis.
- [ ] **Section 5:** Description of the proposed improvements (mathematical & algorithmic details).
- [ ] **Section 6:** Comparative computational evaluation proving the effectiveness of the improvements.
- [ ] Originality check: all text and code are your own work, with any external references properly cited.

### B. Code & Supporting Archive (`.zip` or `.tgz`; MDDM says one archive together with the final report. Confirm the order with the lecturer.)

- [ ] Source code of all models, generator, heuristics, validator, and experiment scripts (clean and commented).
- [ ] Instances (or exact generator script + parameters/seeds to reproduce them identically).
- [ ] Execution instructions / scripts so the lecturer can reproduce results easily.
- [ ] Detailed experiment outputs (CSV files / logs / spreadsheets).
- [ ] Confirm any Git repository remains private.

### C. Oral Exam Preparation

- [ ] Submit the official course evaluation on `esami.unipi.it` **before** the oral exam.
- [ ] Agree on an oral exam date with the lecturer once both report and code archive are approved.
- [ ] Review key insights from your project (why specific formulations/valid inequalities/CG work, LP relaxation strength, symmetry, complexity) and how they connect to the broader course syllabus.
