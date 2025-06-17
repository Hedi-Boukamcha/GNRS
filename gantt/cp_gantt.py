import matplotlib.pyplot as plt

from models.instance import Instance, MathInstance
from conf import *

def plot_gantt_chart(tasks, instance_file: str):
    fig, ax   = plt.subplots(figsize=(10, 3))
    level_idx = {lvl: i for i, lvl in enumerate(CP_GANTT_LEVELS)}
    

    for task in tasks:
        y     = level_idx[task["level"]] 
        rot   = 90 if task["label"].startswith(("M", "Pos")) else 0
        w     = task["duration"]
        x_txt = task["start"] + w / 2          
        y_txt = y + 0.8 / 2  

        ax.barh(y, task["duration"], left=task["start"], color=task["color"], edgecolor='black', hatch=task.get("hatch", None), align="edge")
        ax.text(x_txt, y_txt, task["label"], rotation=rot, ha="center", va="center", fontsize=8, fontweight="bold" if task["label"] in {"L", "M", "Pos"} else "normal")
        time_points = sorted(set([task["start"] for task in tasks] + [task["end"] for task in tasks]))
    t_min = min(t["start"] for t in tasks)
    t_max = max(t["end"]   for t in tasks)
    for i in range(len(CP_GANTT_LEVELS)):
        ax.hlines(i, xmin=t_min, xmax=t_max, colors="grey", linestyles=":", linewidth=0.8, alpha=0.6, zorder=4, clip_on=False)
    ax.set_xticks(time_points)
    ax.set_xticklabels([f'{t:.1f}' for t in time_points], rotation=45, fontsize=8)
    ax.set_xlim(0, max(time_points) + 1)
    ax.set_xlabel("Temps")
    ax.set_yticks(range(len(CP_GANTT_LEVELS))) 
    ax.set_yticklabels(CP_GANTT_LEVELS, fontsize=8)
    ax.set_title(f"Diagramme de Gantt - {instance_file}")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.show()


def cp_gantt(instance: Instance, i: MathInstance, solver, instance_file: str):
    print("\n--- Simulation Gantt à partir du modèle mathématique ---")
    tasks      = []
    for j, job in enumerate(instance.jobs):
        assigned_station = None
        for c in [STATION_1, STATION_2, STATION_3]:
            if solver.BooleanValue(i.s.job_loaded[j][c]):
                assigned_station = c
                break
        if assigned_station is None:
            assigned_station = STATION_1
        for o, op in enumerate(job.operations):
            is_removed: bool = False
            modeB = solver.BooleanValue(i.s.exe_mode[j][o][1])
            for c in i.loop_stations():
                is_removed = is_removed or solver.BooleanValue(i.s.job_unload[j][c])
            has_pos_j = modeB and (o>0 or is_removed or not i.job_modeB[j])
            pos_duration = i.pos_j[j] if has_pos_j else 0
            start = solver.Value(i.s.exe_start[j][o])
            start_move = start - M
            start_pos  = start
            start_weld = start_pos + pos_duration
            duration_weld = i.welding_time[j][o]
            end_weld = start + duration_weld
            if op.type == MACHINE_2:
                mode_str = "Mode C"
            else:
                mode_str = "Mode B" if modeB else "Mode A"
            machine_level = "Machine 2" if op.type == MACHINE_2 else "Machine 1"
            
            #  0) Display processing time 
            tasks.append({
                "label"   : f"J{j+1}_Op{o+1}_({mode_str})",
                "start"   : start_weld,
                "end"     : end_weld,
                "duration": duration_weld,
                "color"   : JOB_COLORS[j % len(JOB_COLORS)],
                "station" : assigned_station,
                "level"   : machine_level,
            })
            # 1) Display (M)
            tasks.append({
                "label"   : f"M",
                "start"   : start_move,
                "end"     : start,
                "duration": M,
                "color"   : JOB_COLORS[j % len(JOB_COLORS)],               
                "level"   : machine_level,
            })

            # 2) Display (Pos) if mode B 
            if pos_duration:
                tasks.append({
                    "label"   : "Pos",
                    "start"   : start_pos,
                    "end"     : start_pos + pos_duration,
                    "duration": pos_duration,
                    "color"   : JOB_COLORS[j % len(JOB_COLORS)],   
                    "level"   : machine_level,        
                })
            
            load_start = solver.Value(i.s.entry_station_date[j][assigned_station])
            station_level = f"Station {assigned_station + 1}"

            # 3) Display L if t=0 (initial loads)
            if load_start == 0:                      
                station_level = f"Station {assigned_station + 1}"
                tasks.append({
                    "label"   : "L",                  
                    "start"   : 0,
                    "end"     : 0.1,                
                    "duration": 1,
                    "color"   : JOB_COLORS[j % len(JOB_COLORS)],
                    "level"   : station_level,
                    "hatch"   : "///"               
                })
            else:
            # 4) Display L if t!=0
                tasks.append({
                    "label"   : "L",               
                    "start"   : load_start,
                    "end"     : load_start + 0.01,      
                    "duration": L,
                    "color"   : JOB_COLORS[j % len(JOB_COLORS)],
                    "level"   : station_level,
                })
    plot_gantt_chart(tasks, instance_file)
