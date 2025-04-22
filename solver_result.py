

import csv
import os

from model import Instance, MathInstance


def save_solution_to_csv(instance: Instance, i:MathInstance, solver, instance_type, num_instance):

    folder_path = os.path.join('data', 'solutions', instance_type)
    os.makedirs(folder_path, exist_ok=True)

    filename = os.path.join(folder_path, f"solution_{instance_type}_{num_instance}.csv")

    headers = [
        "job_id", "operation_id", "big", "due_date", "pos_time", "status", "blocked",
        "op_type", "processing_time",
        "exe_mode", "exe_start", "delay"
    ]

    rows = []
    for j, job in enumerate(instance.jobs):
        for o, op in enumerate(job.operations):
            exe_mode = None
            for m in i.loop_modes():
                if solver.BooleanValue(i.s.exe_mode[j][o][m]):
                    exe_mode = m
            exe_start = solver.Value(i.s.exe_start[j][o])
            delay = solver.Value(i.s.delay[j])
            rows.append([
                j, o, job.big, job.due_date, job.pos_time, job.status, job.blocked,
                op.type, op.processing_time,
                exe_mode, exe_start, delay
            ])
    
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)
    
        print(f"Solution sauvegardée dans : {filename}")
