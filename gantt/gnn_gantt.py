import matplotlib.pyplot as plt
from conf import *
from models.state import *



def gnn_gantt(state: State, instance: str):
    tasks = []
    job_colors = ['#8dd3c7', '#80b1d3', '#fb8072', '#fdb462', '#b3de69', '#fccde5']
    for job in state.job_states:
        color = job_colors[job.id % len(job_colors)]
        for event in job.calendar.events:
            if event.event_type == EXECUTE:
                label = f"P{job.id+1} op{event.operation.id+1} - {event.operation.operation.type} - station {event.station.id+1}"
                tasks.append({
                    "label": label,
                    "start": event.start,
                    "end": event.end,
                    "duration": event.end - event.start,
                    "color": color,
                    "proc_type": event.operation.operation.type 
                })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    for task in tasks:
        level = 0 if task["proc_type"] == MACHINE_1 else 1
        ax.barh(level, task["duration"], left=task["start"], color=task["color"], edgecolor='black')
        ax.text(task["start"] + 0.2, level, f'{task["label"]}', va='bottom', fontsize=7)

    time_points = sorted(set([t["start"] for t in tasks] + [t["end"] for t in tasks]))
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points], rotation=45, fontsize=8)
    ax.set_xlim(0, max(time_points) + 1)
    ax.set_xlabel("Temps")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Procédé 1", "Procédé 2"])
    ax.set_title(f"Diagramme de Gantt - {instance}")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.show()
