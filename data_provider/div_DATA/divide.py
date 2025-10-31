import json
import os


score_file = "all_id_scores.json"   
data_file = "train_ETTH1.jsonl"  
output_high = "High_20p.jsonl"
output_low = "Low_20p.jsonl" 


with open(score_file, "r", encoding="utf-8") as f:
    scores = json.load(f)

id_mae_list = [(str(k), v["mse_diff_avg"]) for k, v in scores.items()]


id_mae_list.sort(key=lambda x: x[1], reverse=True)


n = len(id_mae_list)
top_k = max(1, n // 10)
high_ids = set([x[0] for x in id_mae_list[:top_k]])
low_ids = set([x[0] for x in id_mae_list[-top_k:]])


high_records, low_records = [], []
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        obj_id = str(obj.get("id"))
        if obj_id in high_ids:
            high_records.append(obj)
        elif obj_id in low_ids:
            low_records.append(obj)

with open(output_high, "w", encoding="utf-8") as fh:
    for r in high_records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(output_low, "w", encoding="utf-8") as fl:
    for r in low_records:
        fl.write(json.dumps(r, ensure_ascii=False) + "\n")

