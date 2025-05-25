import random
from models.instance import Instance, MathInstance, PROCEDE_1, PROCEDE_2
import matplotlib.pyplot as plt

def plot_gantt_chart(tasks, instance_file: str):
    fig, ax = plt.subplots(figsize=(10, 3))
    for task in tasks:
        level = 0 if task["proc_type"] == PROCEDE_1 else 1
        ax.barh(level, task["duration"], left=task["start"], color=task["color"], edgecolor='black')
        ax.text(task["start"] + 0.2, level, f'{task["label"]}', va='bottom', fontsize=8, color='black')
        time_points = sorted(set([task["start"] for task in tasks] + [task["end"] for task in tasks]))
       
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points], rotation=45, fontsize=8)

    ax.set_xlim(0, max(time_points) + 1)
    ax.set_xlabel("Temps")
    ax.set_yticks([0, 1]) 
    ax.set_yticklabels(["Procédé 1", "Procédé 2"])
    ax.set_title(f"Diagramme de Gantt - {instance_file}")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.show()

def get_station_from_operation(op):
    # Hypothèse : mapping du type de procédé à une station
    if op.type == PROCEDE_1:
        return 0  # STATION_1
    elif op.type == PROCEDE_2:
        return 1  # STATION_2
    else:
        return 2  # STATION_3 par défaut

def gantt_cp_solution(instance: Instance, i: MathInstance, solver, instance_file: str):
    print("\n--- Simulation Gantt à partir du modèle mathématique ---")
    tasks = []
    job_colors = ['#8dd3c7', '#80b1d3', '#fb8072', '#fdb462', '#b3de69', '#fccde5']
    
    for j, job in enumerate(instance.jobs):
        assigned_station = None
        for c in [0, 1, 2]:
            if solver.BooleanValue(i.s.job_loaded[j][c]):
                assigned_station = c
                break
        if assigned_station is None:
            assigned_station = 0

        for o, op in enumerate(job.operations):
            is_removed: bool = False
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])
            for c in i.loop_stations():
                is_removed = is_removed or solver.BooleanValue(i.s.job_unload[j][c])
            has_pos_j = modeB and (o>0 or is_removed or not i.job_modeB[j])
            start = solver.Value(i.s.exe_start[j][o])
            duration = i.welding_time[j][o] + (i.pos_j[j] if has_pos_j else 0)
            end = start + duration
            station = f"n°{assigned_station + 1}"
            if op.type == 2:
                mode_str = "Mode C"
            else:
                mode_str = "Mode B" if modeB else "Mode A"
            proc_type = 1 if op.type == 2 else 0
            tasks.append({
                "label": f"J{j+1} O{o+1} S{station} ({mode_str})",
                "start": start,
                "end": end,
                "duration": duration,
                "color": job_colors[j % len(job_colors)],
                "station": assigned_station,
                "proc_type": proc_type
            })

    plot_gantt_chart(tasks, instance_file)

def simulate_instance(instance: Instance, i: MathInstance, solver, instance_file: str):
    """
    Simulate and visualize the CP solution for the given instance using matplotlib.
    This function is compatible with the solver_per_file logic and assumes a solved CP model.
    """
    print("\n--- Simulation des dates d'exécution (Gantt Chart) ---")
    tasks = []
    job_colors = ['#8dd3c7', '#80b1d3', '#fb8072', '#fdb462', '#b3de69', '#fccde5']
    
    for j, job in enumerate(instance.jobs):
        for o, op in enumerate(job.operations):
            start = solver.Value(i.s.exe_start[j][o])
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])
            duration = i.welding_time[j][o] + (i.pos_j[j] if modeB else 0)
            end = start + duration
            s = get_station_from_operation(op)
            station = f"n°{s + 1}"
            mode_str = "Mode B" if modeB else "Mode A"

            tasks.append({
                "label": f"J{j+1} O{o+1} S{station} ({mode_str})",
                "start": start,
                "end": end,
                "duration": duration,
                "color": job_colors[j % len(job_colors)]
            })
    plot_gantt_chart(tasks, instance_file)

def simulate_schedule(instance: Instance, i: MathInstance, solver, instance_type, num_instance):
    tasks = []

    for j, job in enumerate(instance.jobs):
        for o, op in enumerate(job.operations):
            start = solver.Value(i.s.exe_start[j][o])
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])  # mode B = index 1
            duration = i.welding_time[j][o] + (i.pos_j[j] if modeB else 0)
            end = start + duration

            tasks.append({
                "label": f"Job {j} - Op {o} (Mode {'B' if modeB else 'A/C'})",
                "start": start,
                "end": end,
                "duration": duration
            })
    plot_gantt_chart(tasks, num_instance)
