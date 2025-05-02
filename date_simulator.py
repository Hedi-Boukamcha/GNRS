import random
from model import Instance, MathInstance, PROCEDE_1, PROCEDE_2
import matplotlib.pyplot as plt

def plot_gantt_chart(tasks, instance_file: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, task in enumerate(reversed(tasks)):
        ax.barh(idx, task["duration"], left=task["start"], color=task["color"], edgecolor='black')
        ax.text(task["start"] + 0.2, idx, f'{task["label"]}', va='center', fontsize=9, color='black')
        ax.vlines([task["start"], task["end"]], ymin=-1, ymax=len(tasks), linestyles='dotted', color='gray')

    ax.set_xlabel("Temps")
    ax.set_yticks([])
    ax.set_title(f"Diagramme de Gantt - {instance_file}")
    plt.tight_layout()
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
    
    job_to_station = {}  # job index -> station index
    for j, job in enumerate(instance.jobs):
        # Assign a single station per job based on its size
        if job.big == 1:
            job_to_station[j] = 1  # Station 2
        else:
            job_to_station[j] = random.choice([0, 1, 2])  # petite pièce → aléatoire

    for j, job in enumerate(instance.jobs):
        assigned_station = job_to_station[j]
        for o, op in enumerate(job.operations):
            start = solver.Value(i.s.exe_start[j][o])
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])
            duration = i.welding_time[j][o] + (i.pos_j[j] if modeB else 0)
            end = start + duration
            station_str = f"n°{assigned_station + 1}"
            mode_str = "Mode B" if modeB else "Mode A"

            tasks.append({
                "label": f"J{j+1} O{o+1} S{station_str} ({mode_str})",
                "start": start,
                "end": end,
                "duration": duration,
                "color": job_colors[j % len(job_colors)],
                "station": assigned_station
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
