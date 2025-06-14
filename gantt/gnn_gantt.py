import matplotlib.pyplot as plt
from conf import *
from models.state import *



def gnn_gantt(state: State, instance: str):
    tasks = []
    job_colors = [
    '#bbe5dd',  # turquoise très pâle
    '#b3d0e5',  # bleu pastel doux
    '#fdb3aa',  # corail clair
    '#fed2a1',  # abricot clair
    '#d1eba5',  # vert anis léger
    '#fde1ef'   # rose très pâle
    ]
    machine_colors   = {MACHINE_1: '#1B9E77', MACHINE_2: '#D95F02'}
    for job in state.job_states:
        color = job_colors[job.id % len(job_colors)]
        for event in job.calendar.events:
            if event.event_type == EXECUTE:
                label = f"P{job.id+1} op{event.operation.id+1} - {event.operation.operation.type}"
                tasks.append({
                    "label": label,
                    "start": event.start,
                    "end": event.end,
                    "duration": event.end - event.start,
                    "color": color,
                    "proc_type": event.operation.operation.type,
                    "station"  : event.station.id+1,          # ← NOUVEAU
                    "job_id"   : job.id   
                })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))

    for task in tasks:
        level   = task["station"]               # S0-S2 sur l’axe Y
        facecol = job_colors[task["job_id"] % len(job_colors)]
        edgecol = machine_colors[task["proc_type"]]

        ax.barh(level, task["duration"], left=task["start"],
                color=facecol, edgecolor=edgecol, linewidth=2)

        ax.text(task["start"] + 0.5, level, task["label"],
                va='center', fontsize=7, color='black')

    # ─── axes et légende ─────────────────────────────────────────────────
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['0','S1', 'S2', 'S3', ''])
    time_points = sorted({t["start"] for t in tasks}.union({t["end"] for t in tasks}))
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points], rotation=45, fontsize=8)
    ax.set_xlim(0, max(time_points) + 5)
    ax.set_xlabel("Temps")
    ax.set_title(f"Diagramme de Gantt – {instance}")

    # légende procédés (bordure)
    handles = [plt.Rectangle((0,0), 1, 1, fc='white',
                            ec=machine_colors[p], lw=2) for p in sorted(machine_colors)]
    ax.legend(handles, [f"Machine {p+1}" for p in sorted(machine_colors)],
            loc='upper right', fontsize=7)

    ax.grid(axis='x', linestyle='--', alpha=.3)
    plt.tight_layout()
    plt.show()
