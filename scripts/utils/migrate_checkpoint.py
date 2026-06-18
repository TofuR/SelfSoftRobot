#!/usr/bin/env python
"""migrate_checkpoint.py — Checkpoint 格式迁移工具。

解决 S1 优化 (GRUCell → nn.GRU) 导致的 checkpoint 不兼容问题：

用法:
    python scripts/utils/migrate_checkpoint.py \\
        --input train_log/gt_transition/exp_20260616_3/phase_gt_transition/model/best_model.pt \\
        --output train_log/gt_transition/exp_20260616_3/phase_gt_transition/model/best_model_migrated.pt
"""

import argparse
import torch
from pathlib import Path


def migrate_grucell_to_gru(state_dict):
    """迁移 GRUCell state_dict 到 nn.GRU 格式。

    GRUCell 键:
        gru.weight_ih  → gru.weight_ih_l0
        gru.weight_hh  → gru.weight_hh_l0
        gru.bias_ih    → gru.bias_ih_l0
        gru.bias_hh    → gru.bias_hh_l0

    权重 shape 一致，只需添加 _l0 后缀。

    Args:
        state_dict: 原始 state_dict (GRUCell 格式)

    Returns:
        新 state_dict (nn.GRU 格式)
    """
    new_dict = {}
    gru_renamed = False

    for key, value in state_dict.items():
        if key in ('gru.weight_ih', 'gru.weight_hh',
                   'gru.bias_ih', 'gru.bias_hh'):
            # 添加 _l0 后缀（nn.GRU 单层格式）：gru.weight_ih → gru.weight_ih_l0
            new_key = key + '_l0'
            new_dict[new_key] = value
            gru_renamed = True
        else:
            new_dict[key] = value

    if gru_renamed:
        print(f"  [迁移] GRUCell → nn.GRU 格式")
        print(f"    旧键: gru.weight_ih/hh/bias_ih/bias_hh")
        print(f"    新键: gru.weight_ih_l0/hh_l0/bias_ih_l0/bias_hh_l0")

    return new_dict


def migrate_gru_to_grucell(state_dict):
    """迁移 nn.GRU state_dict 到 GRUCell 格式（逆操作）。

    Args:
        state_dict: nn.GRU 格式 state_dict

    Returns:
        GRUCell 格式 state_dict
    """
    new_dict = {}
    gru_renamed = False

    for key, value in state_dict.items():
        if key.startswith('gru.') and '_l0' in key:
            # 移除 _l0 后缀
            new_key = key.replace('_l0.', '')
            new_dict[new_key] = value
            gru_renamed = True
        else:
            new_dict[key] = value

    if gru_renamed:
        print(f"  [迁移] nn.GRU → GRUCell 格式")

    return new_dict


def detect_format(state_dict):
    """检测 state_dict 是 GRUCell 还是 nn.GRU 格式。"""
    gru_keys = [k for k in state_dict.keys() if k.startswith('gru.')]
    if not gru_keys:
        return 'none'

    # 检查是否有 _l0 后缀
    has_l0 = any('_l0' in k for k in gru_keys)

    if has_l0:
        return 'gru'
    elif any(k in ('gru.weight_ih', 'gru.weight_hh') for k in gru_keys):
        return 'grucell'
    else:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Checkpoint 格式迁移工具')
    parser.add_argument('--input', '-i', required=True, help='输入 checkpoint 路径')
    parser.add_argument('--output', '-o', required=True, help='输出 checkpoint 路径')
    parser.add_argument('--direction', '-d', default='auto',
                        choices=['auto', 'grucell_to_gru', 'gru_to_grucell'],
                        help='迁移方向（auto 自动检测）')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return

    print(f"加载 checkpoint: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')

    # 检测格式
    detected = detect_format(state_dict)
    print(f"  检测格式: {detected}")

    # 确定迁移方向
    if args.direction == 'auto':
        if detected == 'grucell':
            direction = 'grucell_to_gru'
        elif detected == 'gru':
            direction = 'gru_to_grucell'
        else:
            print(f"  跳过: 未检测到 GRU 相关键")
            direction = 'none'
    else:
        direction = args.direction

    # 执行迁移
    if direction == 'grucell_to_gru':
        new_state_dict = migrate_grucell_to_gru(state_dict)
    elif direction == 'gru_to_grucell':
        new_state_dict = migrate_gru_to_grucell(state_dict)
    else:
        new_state_dict = state_dict

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_path)
    print(f"保存到: {output_path}")

    # 验证
    print("\n验证加载:")
    verify = torch.load(output_path, map_location='cpu')
    verify_format = detect_format(verify)
    print(f"  输出格式: {verify_format}")
    print(f"  参数数量: {len(verify)}")


if __name__ == '__main__':
    main()
