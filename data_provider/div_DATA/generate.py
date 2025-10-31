import pandas as pd
import json
import random

def generate_windows(ot_series, window_size=100, num_windows=1000, start_index=0):
    windows = []
    max_start = len(ot_series) - window_size
    for i in range(start_index, min(start_index + num_windows, max_start + 1)):
        window = ot_series[i:i+window_size].tolist()
        windows.append({
            "sequence": window,
            "id": i  # 起始行号
        })
    return windows

# -------------------------
# 读取数据，生成滑动窗口
# -------------------------
df = pd.read_csv("/Users/l/Desktop/div_DATA/origin_data/ETTh1.csv")  
ot_series = df['OT'].values
windows_list = generate_windows(ot_series, window_size=100, num_windows=1000, start_index=0)

# 保存完整 JSONL 文件
with open("ETTH1.jsonl", "w") as f:
    for w in windows_list:
        f.write(json.dumps(w) + "\n")

# -------------------------
# 按比例划分 train/val/last
# -------------------------
total = len(windows_list)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.2)

train_list = windows_list[:train_end]
val_list = windows_list[train_end:val_end]
last_list = windows_list[val_end:]

# 保存划分好的数据为 JSONL 文件
def save_jsonl(filename, data_list):
    with open(filename, "w") as f:
        for w in data_list:
            f.write(json.dumps(w) + "\n")

save_jsonl("train_ETTH1.jsonl", train_list)
save_jsonl("val_ETTH1.jsonl", val_list)
save_jsonl("last_ETTH1.jsonl", last_list)

# -------------------------
# 将 train 集合随机划分成 5 份
# -------------------------
random_seed = 42
random.Random(random_seed).shuffle(train_list)
split_size = len(train_list) // 5

for i in range(5):
    split = train_list[i*split_size : (i+1)*split_size]
    save_jsonl(f"ETO{i+1}.jsonl", split)

print("JSONL 数据生成及划分完成！")