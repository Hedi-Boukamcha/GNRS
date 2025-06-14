from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from conf import *
from models.state import *


def gnn_gantt(state: State, instance: str):
    job_colors = ['#bbe5dd', "#c6def0", "#f7cbc5",
                  '#fed2a1', '#d1eba5', '#fde1ef']
    machine_colors = {MACHINE_1: "#00FFF2", MACHINE_2: "#FF0000"}

    # ─── construire tasks ────────────────────────────────────────────
    tasks = []
    for job in state.job_states:
        for e in job.calendar.events:
            if e.event_type == EXECUTE:
                tasks.append({
                    "label"    : f"Job {job.id+1} → Op {e.operation.id+1}",
                    "start"    : e.start,
                    "duration" : e.end - e.start,
                    "end"      : e.end,
                    "station"  : e.station.id+1,     # 0–2
                    "job_id"   : job.id,
                    "proc_type": e.operation.operation.type
                })

    # ─── tracer ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    renderer = fig.canvas.get_renderer()

    for t in tasks:
        level = t["station"]
        face  = job_colors[t["job_id"] % len(job_colors)]
        edge  = machine_colors[t["proc_type"]]

        # barre
        ax.barh(level, t["duration"], left=t["start"],
                color=face, edgecolor=edge, linewidth=2)

        # calcul police auto
        bar_px = (ax.transData.transform([[t["start"], 0],
                                          [t["start"] + t["duration"], 0]])[1, 0]
                  - ax.transData.transform([[0, 0]])[0, 0])

        for fs in range(10, 3, -1):
            test = ax.text(0, 0, t["label"], fontsize=fs)
            w = test.get_window_extent(renderer).width
            test.remove()
            if w < bar_px - 4:      # marge interne
                break

        # étiquette finale (une seule)
        ax.text(t["start"] + t["duration"]/2, level,
                t["label"], ha='center', va='center',
                fontsize=fs, clip_on=True)

    # ─── axes et légende (inchangés) ──────────────────────────────────
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['0','S1', 'S2', 'S3', ''])

    time_points = sorted({t["start"] for t in tasks}
                         | {t["end"] for t in tasks})
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points],
                       rotation=45, fontsize=8)
    ax.set_xlim(0, max(time_points) + 5)

    ax.set_xlabel("Temps")
    ax.set_title(f"Diagramme de Gantt – {instance}")

    handles = [plt.Rectangle((0,0), 1, 1, fc='white',
                             ec=machine_colors[p], lw=2)
               for p in sorted(machine_colors)]
    ax.legend(handles, [f"Machine {p+1}" for p in sorted(machine_colors)],
              loc='upper right', fontsize=7)

    ax.grid(axis='x', linestyle='--', alpha=.3)
    plt.tight_layout()
    plt.show()
