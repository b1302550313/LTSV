#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import argparse
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset
from time_moe.datasets.pre_split_dataset import PreSplitEvalDataset
import tempfile

# ------------------ 工具函数 ------------------
def split_sequence_to_jsonl(input_json_path, context_length, prediction_length):
    """
    将每条长序列划分成 context + prediction 的窗口
    生成临时 JSONL 文件用于模型评估
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False)
    with open(input_json_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            seq = data['sequence']
            seq_id = data.get('id', line_num)

            total_len = context_length + prediction_length
            # 滑动窗口切分序列
            for start_idx in range(0, len(seq) - total_len + 1):
                window = seq[start_idx:start_idx+total_len]
                json_obj = {
                    'inputs': window[:context_length],
                    'labels': window[context_length:],
                    'id': f"{seq_id}_{start_idx}"
                }
                temp_file.write(json.dumps(json_obj) + "\n")
    temp_file.flush()
    return temp_file.name

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    """
    初始化 NCCL 分布式通信（多 GPU 环境）
    """
    dist.init_process_group(
        "nccl",
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

def count_num_tensor_elements(tensor):
    """
    返回 tensor 元素总数
    """
    n = 1
    for s in tensor.shape:
        n *= s
    return n

# ------------------ 指标类 ------------------
class SumEvalMetric:
    """
    基类：累积计算指标
    """
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass

class MSEMetric(SumEvalMetric):
    """均方误差"""
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)

class MAEMetric(SumEvalMetric):
    """平均绝对误差"""
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))

# ------------------ 模型封装 ------------------
class TimeMoE:
    """
    封装 Time-MoE 模型
    支持：
    1. GPU / CPU
    2. 归一化和反归一化处理
    """
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            # 使用自定义模型
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype='auto',
            )
        except:
            # fallback: HuggingFace AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype='auto',
                trust_remote_code=True,
            )
        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')
        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch, ot_mean=None, ot_std=None):
        """
        对 batch 进行预测
        支持输入归一化和输出反归一化
        """
        inputs = batch['inputs'].to(self.device).to(self.model.dtype)
        labels = batch['labels'].to(self.device)
        # 如果提供了 mean/std，则对输入做归一化
        if ot_mean is not None and ot_std is not None:
            inputs_norm = (inputs - ot_mean) / ot_std
        else:
            inputs_norm = inputs

        # 模型生成预测
        outputs = self.model.generate(
            inputs=inputs_norm,
            max_new_tokens=self.prediction_length,
        )
        preds = outputs[:, -self.prediction_length:]
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]

        # 反归一化
        if ot_mean is not None and ot_std is not None:
            preds = preds * ot_std + ot_mean
        return preds, labels

# ------------------ 评估函数 ------------------
def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length

    # ---------------- 分布式 GPU 设置 ----------------
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        try:
            setup_nccl(rank=rank, world_size=world_size)
            device = f"cuda:{local_rank}"
            is_dist = True
        except:
            device = 'cpu'
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # ---------------- 划分序列 ----------------
    temp_jsonl_path = split_sequence_to_jsonl(args.data, context_length, prediction_length)
    dataset = PreSplitEvalDataset(temp_jsonl_path)

    # 分布式采样器
    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None

    test_dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         shuffle=False, num_workers=2, prefetch_factor=2, drop_last=False)

    # ---------------- 计算整个数据集的均值和标准差 ----------------
    sequences = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            seq = json.loads(line)['sequence']
            sequences.extend(seq)
    sequences = torch.tensor(sequences, dtype=torch.float32)
    ot_mean = sequences.mean()
    ot_std = sequences.std()
    print(f"OT mean={ot_mean:.4f}, std={ot_std:.4f}")

    # ---------------- 初始化指标与模型 ----------------
    metric_list = [MSEMetric(name='mse'), MAEMetric(name='mae')]
    model = TimeMoE(args.model, device, context_length, prediction_length)

    # ---------------- 遍历数据集进行预测 ----------------
    acc_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch, ot_mean=ot_mean, ot_std=ot_std)
            # 更新指标
            for metric in metric_list:
                metric.push(preds, labels)
            acc_count += count_num_tensor_elements(preds)

    # ---------------- 输出最终结果 ----------------
    ret_metric = {metric.name: metric.value / acc_count for metric in metric_list}
    print(f'{rank} - {ret_metric}')

# ------------------ 主函数 ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument('--model', '-m', type=str, default='Maple728/TimeMoE-50M')
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--context_length', '-c', type=int)
    parser.add_argument('--prediction_length', '-p', type=int, default=96)
    args = parser.parse_args()
    if args.context_length is None:
        # 默认 context 长度为 prediction_length 的 4 倍
        args.context_length = args.prediction_length * 4
    evaluate(args)
