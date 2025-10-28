#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
ILL_main.py
作用：
1. 基础模型评估
2. 单样本微调+评分
3. 实时写入分数和时间日志
4. 最终生成 H/L 数据集
"""

import os
import json
import tempfile
import subprocess
import re
from statistics import mean
import argparse

# ================= 参数 =================
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="Maple728/TimeMoE-50M")
parser.add_argument("--data_dir", type=str, default="/root/Time-MoE/WEA")
parser.add_argument("--output_dir", type=str, default="/root/Time-MoE/WEA_R")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--context_length", type=int, default=76)
parser.add_argument("--prediction_length", type=int, default=24)
parser.add_argument("--datasets", nargs='+', default=[
    "WEA1.jsonl",
    "WEA2.jsonl",
    "WEA3.jsonl",
    "WEA4.jsonl",
    "WEA5.jsonl"
])
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

result_file = os.path.join(args.output_dir, "eval_log.txt")
all_scores_file = os.path.join(args.output_dir, "all_id_scores.json")
final_H_file = os.path.join(args.output_dir, "H.jsonl")
final_L_file = os.path.join(args.output_dir, "L.jsonl")
time_log_file = os.path.join(args.output_dir, "time_log.json")

# ================= 工具函数 =================
def parse_metrics(output_str):
    mse = re.search(r"mse.*?([\d.\-]+)", output_str.lower())
    mae = re.search(r"mae.*?([\d.\-]+)", output_str.lower())
    return (float(mse.group(1)) if mse else None,
            float(mae.group(1)) if mae else None)

def save_all_scores(scores_dict):
    with open(all_scores_file, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2, ensure_ascii=False)

def save_time_log(log_dict):
    with open(time_log_file, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)

# ================= Step 1: 基础模型评估 =================
print("[Base Model] Evaluating on all validation sets...")
base_scores = {}
for ds in args.datasets:
    path = os.path.join(args.data_dir, ds)
    try:
        result = subprocess.run(
            [
                "python", "run_eval.py",
                "-m", args.base_model,
                "-d", path,
                "--batch_size", str(args.batch_size),
                "--context_length", str(args.context_length),
                "--prediction_length", str(args.prediction_length)
            ],
            capture_output=True, text=True, check=True
        )
        mse, mae = parse_metrics(result.stdout)
        base_scores[ds] = (mse, mae)
        with open(result_file, "a", encoding="utf-8") as rf:
            rf.write(f"[Base Model] {ds}: mse={mse}, mae={mae}\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! Base model eval failed on {ds}")
        print(e.stdout)
        print(e.stderr)

# ================= Step 2: 微调和评分 =================
id_scores = {}
time_log = {}

for i, train_file in enumerate(args.datasets):
    train_path = os.path.join(args.data_dir, train_file)
    val_files = [f for f in args.datasets if f != train_file]

    print(f"\n[Fold {i+1}] Using {train_file} for fine-tuning...")

    with open(train_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                json_obj = json.loads(line)
                sample_id = json_obj.get("id", f"{train_file}_{line_num}")
            except json.JSONDecodeError:
                print(f"!!! JSON decode failed in {train_file} line {line_num}, skipping...")
                continue

            with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as temp_data_file:
                temp_data_file.write(line.strip() + "\n")
                temp_data_path = temp_data_file.name

            fold_diffs = []

            try:
                with tempfile.TemporaryDirectory() as temp_model_path:
                    # ------------------ 微调 ------------------
                    import time
                    start_ft = time.time()
                    try:
                        subprocess.run([
                            "python", "main.py",
                            "-d", temp_data_path,
                            "-m", args.base_model,
                            "--output_path", temp_model_path,
                            "--num_train_epochs", "1",
                            "--save_strategy", "no"
                        ], check=True)
                    except subprocess.CalledProcessError:
                        print(f"!!! Fine-tune failed for {sample_id}")
                        continue
                    end_ft = time.time()
                    fine_tune_time = round(end_ft - start_ft, 3)

                    # ------------------ 五折评分 ------------------
                    start_sc = time.time()
                    for val_file in val_files:
                        val_path = os.path.join(args.data_dir, val_file)
                        try:
                            result = subprocess.run(
                                [
                                    "python", "run_eval.py",
                                    "-m", temp_model_path,
                                    "-d", val_path,
                                    "--batch_size", str(args.batch_size),
                                    "--context_length", str(args.context_length),
                                    "--prediction_length", str(args.prediction_length)
                                ],
                                capture_output=True, text=True, check=True
                            )
                            fine_mse, fine_mae = parse_metrics(result.stdout)
                            base_mse, base_mae = base_scores[val_file]
                            if fine_mse is not None and fine_mae is not None:
                                fold_diffs.append({
                                    "val_set": val_file,
                                    "mse_diff": base_mse - fine_mse,
                                    "mae_diff": base_mae - fine_mae
                                })
                        except subprocess.CalledProcessError:
                            print(f"!!! Eval failed for {sample_id} on {val_file}")
                            continue
                    end_sc = time.time()
                    scoring_time = round(end_sc - start_sc, 3)

            finally:
                os.remove(temp_data_path)

            # ------------------ 计算平均分 ------------------
            if fold_diffs:
                avg_mse_diff = mean([x["mse_diff"] for x in fold_diffs])
                avg_mae_diff = mean([x["mae_diff"] for x in fold_diffs])
                id_scores[sample_id] = {
                    "id": sample_id,
                    "fold_scores": fold_diffs,
                    "mse_diff_avg": round(avg_mse_diff, 6),
                    "mae_diff_avg": round(avg_mae_diff, 6)
                }
                with open(result_file, "a", encoding="utf-8") as rf:
                    rf.write(f"ID {sample_id} final_score: {id_scores[sample_id]}\n")
                save_all_scores(id_scores)

                time_log[sample_id] = {
                    "fine_tune_time": fine_tune_time,
                    "scoring_time": scoring_time,
                    "avg_score": round(avg_mse_diff, 6)
                }
                save_time_log(time_log)

# ================= Step 3: 全局平均分 =================
all_mse = [v["mse_diff_avg"] for v in id_scores.values()]
all_mae = [v["mae_diff_avg"] for v in id_scores.values()]
global_mse_avg = mean(all_mse)
global_mae_avg = mean(all_mae)
print(f"\nGlobal Average: mse_diff={global_mse_avg:.6f}, mae_diff={global_mae_avg:.6f}")

# H/L 划分
sorted_scores = sorted(id_scores.values(), key=lambda x: x["mse_diff_avg"], reverse=True)
mid_index = len(sorted_scores) // 2
high_quality = sorted_scores[:mid_index]
low_quality = sorted_scores[mid_index:]

with open(final_H_file, "w", encoding="utf-8") as fh, \
     open(final_L_file, "w", encoding="utf-8") as fl:
    for v in high_quality:
        fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    for v in low_quality:
        fl.write(json.dumps(v, ensure_ascii=False) + "\n")

print(f">> 高质量数据保存到 {final_H_file}")
print(f">> 低质量数据保存到 {final_L_file}")