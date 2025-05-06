from ortools.sat.python import cp_model
import itertools, sys

# ────────────────────────────────────────────────────────────
#  Pretty printer for the welding‑robot scheduling model
# ────────────────────────────────────────────────────────────
STATUS_MEANING = ["UNKNOWN", "MODEL_INVALID", "FEASIBLE",
                  "INFEASIBLE", "OPTIMAL"]          # <- keep in sync with CpSolver

def pretty_print_solution(i, solver, stream=sys.stdout):
    """
    Nicely formats the contents of `i.s` (MathInstance solution wrapper)
    after a call to `solver.Solve(model)`.

    Parameters
    ----------
    i       : MathInstance      (holds the data and the `i.s` variable tree)
    solver  : cp_model.CpSolver (already populated with a solution)
    stream  : file‑like         (defaults to sys.stdout; give a file handle to log)
    """

    status_code = solver.Status()
    print("\n" + "=" * 70, file=stream)
    print(f"Solver status : {STATUS_MEANING[status_code]} ({status_code})",
          file=stream)

    # Bail out early if the model is infeasible / invalid
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No feasible solution available.", file=stream)
        print("=" * 70 + "\n", file=stream)
        return

    # -----   Top‑level objective  ------------------------------------------------
    print(f"Objective value     : {solver.ObjectiveValue():.0f}", file=stream)
    print(f"   – C_max          : {solver.Value(i.s.C_max)}",      file=stream)
    print("-" * 70, file=stream)

    # -----   Helper lambdas  -----------------------------------------------------
    get_loaded_station = lambda j: next(                 # first station with True
        (c for c in i.loop_stations()
         if solver.BooleanValue(i.s.job_loaded[j][c])),  # BooleanValue() nicer for bools
        None)

    get_mode = lambda j, o: next(
        (m for m in i.loop_modes()
         if solver.BooleanValue(i.s.exe_mode[j][o][m])),
        None)

    # -----   Per‑job details  ----------------------------------------------------
    for j in i.loop_jobs():
        print(f"\nJob {j}", file=stream)
        print("─" * 8, file=stream)
        station = get_loaded_station(j)
        print(f"  • Loaded station      : {station}", file=stream)

        entry_dates = {c: solver.Value(i.s.entry_station_date[j][c])
                       for c in i.loop_stations()}
        print(f"  • Entry dates (per st): {entry_dates}", file=stream)

        print(f"  • Delay               : {solver.Value(i.s.delay[j])}", file=stream)

        # Operations header
        op_header = "      op | start | mode | parallel"
        print(op_header, file=stream)
        print("      " + "-" * (len(op_header) - 6), file=stream)

        for o in i.loop_operations(j):
            start     = solver.Value(i.s.exe_start[j][o])
            mode      = get_mode(j, o)
            parallel  = bool(solver.BooleanValue(i.s.exe_parallel[j][o]))
            print(f"      {o:>2} | {start:>5} | {mode:>4} | {str(parallel):>8}",
                  file=stream)

    # -----   Optional: job unload flags  ----------------------------------------
    unload_lines = []
    for j in i.loop_jobs():
        unload_stations = [c for c in i.loop_stations()
                           if solver.BooleanValue(i.s.job_unload[j][c])]
        if unload_stations:
            unload_lines.append(f"Job {j} unloads from stations: {unload_stations}")

    if unload_lines:
        print("\n" + "-" * 70, file=stream)
        print("\n".join(unload_lines), file=stream)

    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70 + "\n", file=stream)


# ────────────────────────────────────────────────────────────
# Example of use
# ────────────────────────────────────────────────────────────
# model, i.s = init_vars(cp_model.CpModel(), my_instance)
# model, i.s = init_objective_function(model, my_instance)
# ... add constraints ...
# solver     = cp_model.CpSolver()
# status     = solver.Solve(model)
# pretty_print_solution(my_instance, solver)