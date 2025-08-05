import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from conf import *
from models.state import State

# #########################
# =*= DISPLAY GNN GANTT =*=
# #########################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT"

def resource_calendars(state: State):
    return {
        "Station 1" : state.all_stations.get(STATION_1).calendar.events,
        "Station 2" : state.all_stations.get(STATION_2).calendar.events,
        "Station 3" : state.all_stations.get(STATION_3).calendar.events,
        "Machine 1" : state.machine1.calendar.events,
        "Machine 2" : state.machine2.calendar.events,
        "Robot"     : state.robot.calendar.events,
        "Positioner": [e for e in state.machine1.calendar.events if e.event_type == POS],
    }

def gnn_gantt(path: str, state: State, instance: str, bar_h: float = 0.8, min_bar_for_text: float = 5):
    calendars   = resource_calendars(state)
    level_index = {lvl: i for i, lvl in enumerate(GNN_GANTT_LEVELS)}
    tasks       = []

    # 3-a. Construction des tâches
    for lvl, events in calendars.items():
        for e in events:
            dur = e.end - e.start
            if dur < MIN_REAL_DURATION and e.event_type == LOAD:
                continue

            job    = getattr(e, "job", None)
            job_id = job.id if job else -1
            color  = JOB_COLORS[job_id % len(JOB_COLORS)] if job_id >= 0 else "#cccccc"

            if e.event_type in {EXECUTE, HOLD} and job and e.operation:
                op_id = e.operation.id + 1
                para_text: str = " (S)"
                for d in state.decisions:
                    if d.job_id == e.job.id and d.operation_id == e.operation.id and d.parallel:
                        if d.machine == MACHINE_2:
                            para_text = " (P->J"+str(d.comp)+ ")"
                        else:
                            para_text = " (P)"
                label = f"{'execute'+para_text if e.event_type==EXECUTE else 'hold'} : J{job_id+1} ⇒ Op{op_id}"
            else:
                label = EVENT_NAMES[e.event_type]

            tasks.append({
                "level"     : lvl,
                "start"     : e.start,
                "end"       : e.end,
                "dur"       : dur,
                "event_type": e.event_type,
                "label"     : label,
                "color"     : color,
            })

    if not tasks:
        print("Aucun évènement à tracer.")
        return

    # 3-b. Figure et barre de temps
    t_min   = min(t["start"] for t in tasks)
    t_max   = max(t["end"]   for t in tasks)
    fig, ax = plt.subplots(figsize=(max(12, (t_max - t_min) * .12), 6))

    # 3-c. Tracé des barres
    for t in tasks:
        y = level_index[t["level"]]
        ax.barh(y, t["dur"], left=t["start"],
                height=bar_h, align="edge",
                color=t["color"], edgecolor="black")

        rot = 90 if t["label"] in {"move", "load", "unload"} else 0
        fw  = 'bold' if t["event_type"] in BOLD_EVENTS else 'normal'

        if rot == 90 or t["dur"] >= min_bar_for_text:
            x_text, ha = t["start"] + t["dur"]/2, "center"
        else:
            x_text, ha = t["end"] + 0.2, "left"
        ax.text(x_text, y + bar_h/2, t["label"], rotation=rot, ha=ha, va="center", fontsize=7, fontweight=fw)

    # 3-d. Lignes de niveau
    for i in range(len(GNN_GANTT_LEVELS)):
        ax.hlines(i, t_min, t_max, colors="grey", linestyles=":", linewidth=0.8, alpha=0.6, zorder=0)

    # 3-e. Repères « load » initiaux (stations qui ont un LOAD à t_min)
    base_width = 1
    for st in STATIONS:
        events_lvl = calendars[st]
        first_load = next((e for e in events_lvl if e.event_type == LOAD and e.start == t_min), None)
        if not first_load:
            continue
        job    = first_load.job
        job_id = job.id if job else -1
        color  = JOB_COLORS[job_id % len(JOB_COLORS)] if job else "#cccccc"
        y      = level_index[st]

        ax.add_patch(Rectangle((t_min, y), base_width, bar_h, facecolor=color, edgecolor="black", hatch="///", clip_on=False, zorder=3))
        ax.text(t_min + base_width/2, y + bar_h/2, "", rotation=90, ha="center", va="center",zorder=2)

    # 3-f. Axe X
    times = sorted({p for t in tasks for p in (t["start"], t["end"])})
    ax.set_xticks(times)
    ax.set_xticklabels([f"{x:.1f}" for x in times], rotation=45, fontsize=6)
    ax.set_xlim(times[0], times[-1])

    # 3-g. Axe Y
    ax.set_yticks(range(len(GNN_GANTT_LEVELS)))
    ax.set_yticklabels(GNN_GANTT_LEVELS, fontsize=8)

    ax.set_title(f"Diagramme de Gantt – {instance}")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=700, bbox_inches="tight")
    plt.close(fig)