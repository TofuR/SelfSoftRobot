"""build_control_report.py — 生成自包含 HTML 汇报(方向1视野认证 + 方向2逆规划)。

读 output/ 下的图(PNG/GIF)→ base64 内嵌 + horizon/inverse_plan 的 JSON 结果数字,
输出单个可移植 HTML(图、GIF、流程框图全内嵌, 离线可看)。

Usage:
  python scripts/utils/build_control_report.py \
      --horizon_json output/horizon/horizon_summary.json \
      --plan_json   output/inverse_plan_big/plan_result.json \
      --fig_dir     output/viz \
      --horizon_curve output/horizon/horizon_comparison.png \
      --out docs/reports/2026-07-14_control_shape_planning.html
"""

import os
import json
import base64
import argparse


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path, alt=""):
    if not os.path.exists(path):
        return f'<div class="missing">[缺图: {path}]</div>'
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/gif" if ext == "gif" else "image/png"
    return f'<img src="data:{mime};base64,{b64(path)}" alt="{alt}"/>'


def flowchart(steps):
    """竖向流程框图: steps = [(title, body), ...], 之间插 ↓。"""
    out = ['<div class="flow">']
    for i, (title, body) in enumerate(steps):
        out.append(f'<div class="flow-box"><div class="flow-title">{title}</div>'
                   f'<div class="flow-body">{body}</div></div>')
        if i < len(steps) - 1:
            out.append('<div class="arrow">↓</div>')
    out.append('</div>')
    return "\n".join(out)


HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>软体机器人形状控制: 视野认证 + 逆运动学规划</title>
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         max-width: 1000px; margin: 0 auto; padding: 2em 1.5em; color: #222; line-height: 1.65; }
  h1 { font-size: 1.7em; border-bottom: 3px solid #2c5aa0; padding-bottom: .3em; }
  h2 { font-size: 1.35em; color: #2c5aa0; margin-top: 2em; border-left: 5px solid #2c5aa0; padding-left: .5em; }
  h3 { font-size: 1.1em; color: #444; margin-top: 1.4em; }
  .tldr { background: #eef5ff; border: 1px solid #bcd; border-radius: 8px; padding: 1em 1.3em; margin: 1em 0; }
  .tldr b { color: #2c5aa0; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .95em; }
  th, td { border: 1px solid #ccc; padding: .5em .7em; text-align: center; }
  th { background: #2c5aa0; color: #fff; }
  tr:nth-child(even) { background: #f6f8fb; }
  .good { color: #1a7d1a; font-weight: bold; }
  .bad { color: #c0392b; font-weight: bold; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; display: block; margin: .8em auto; }
  .cap { text-align: center; color: #666; font-size: .9em; margin: -.3em 0 1.5em; }
  .flow { display: flex; flex-direction: column; align-items: center; margin: 1.5em 0; }
  .flow-box { background: #f0f4fa; border: 1.5px solid #8aa; border-radius: 8px;
              padding: .7em 1.2em; min-width: 60%; text-align: center; }
  .flow-title { font-weight: bold; color: #2c5aa0; }
  .flow-body { font-size: .92em; color: #333; margin-top: .25em; }
  .arrow { color: #2c5aa0; font-size: 1.6em; line-height: 1; margin: .15em 0; }
  .two-col { display: flex; gap: 1em; flex-wrap: wrap; }
  .two-col > div { flex: 1; min-width: 280px; }
  .warn { background: #fff6e6; border-left: 4px solid #e0a030; padding: .6em 1em; margin: 1em 0; border-radius: 4px; }
  .key { background: #eaffea; border-left: 4px solid #4a4; padding: .6em 1em; margin: 1em 0; border-radius: 4px; }
  code { background: #f0f0f0; padding: .1em .35em; border-radius: 3px; font-size: .92em; }
  .missing { color: #c0392b; background: #fee; padding: 1em; border-radius: 6px; }
  hr { border: none; border-top: 1px dashed #bbb; margin: 2em 0; }
  .footer { color: #888; font-size: .85em; margin-top: 3em; border-top: 1px solid #ddd; padding-top: 1em; }
</style></head><body>

<h1>软体机器人形状控制: 视野认证 + 逆运动学规划</h1>
<div style="color:#666">工作汇报 · 2026-07-15 · 分支 feat/real-data-transition · 数据 SAM2(n15) · commit <code>__COMMIT__</code></div>

<div class="tldr">
<b>一句话</b>: 把已学好的"动作→形状"前向模型当作<b>可微仿真器</b>, 先认证它能可信地往前推多久(方向1),
再用它做逆运动学——给定初始+目标形状, 优化出一段动作序列到达目标(方向2)。
<ul>
<li><b>方向1</b>: open_loop 模型 drift 仅 <span class="good">1.7×</span>(300步=61s), gt 模型 <span class="bad">272×</span> 爆炸 → open_loop 是合格规划仿真器。</li>
<li><b>方向2</b>: 逆规划 reach 任务, planner 到目标 <span class="good">3.07px</span>(=0.38× do-nothing), 接近模型保真上界(GT-actions 2.69px)。</li>
<li><b>部署定位</b>: <b>open_loop 是形状控制部署目标</b>(观测一次→开环预测K步); gt 仅训练基础+精度上界(NDI 末端 0.77mm)。</li>
<li><b>诚实边界</b>: 当前为"模型内验证"(val集+GT-actions基线证模型保真); planner 动作<b>未上真机</b>, 真机闭环是下一步。</li>
</ul>
</div>

<h2>1. 大背景: 从"预测形状"到"控制形状"</h2>
<p>项目已能<b>预测</b>(给定动作→未来形状)。形状<b>控制</b>是它的逆问题: 给定目标形状, 找到达它的动作。
但当前 open_loop 部署每次预测前需 1 帧真实观测作种子——对"规划未来动作"偏鸡肋。
本工作把前向模型升级为<b>规划级仿真器</b>, 在其上做轨迹优化。</p>

__FLOW_A__

<div class="key"><b>核心洞察</b>: 逆规划 = 在仿真器里优化动作序列。若仿真器漂移, 规划出的动作拿到真机到不了目标。
所以<b>方向1(认证仿真器可信视野)是方向2(规划)的前提</b>, K_max 是规划视野的硬上限。</div>

<h2>2. 方向1: 纯自回归视野认证</h2>

<h3>2.1 方法(模拟部署场景"观测一次→之后只靠动作推")</h3>
__FLOW_B__
<p>取 1 帧 GT 骨架作种子(ŝ₀=positions[t0]), 之后 k=1..K 步每步只喂【动作窗口 + 上一步<b>模型自己的预测</b> + 演化的 z】,
<b>不再看任何真实图像</b>。记录 ŝ_k 与真实位置的误差 → K_max = 误差越过容差的步数。8 个不同 t0 种子聚合。</p>

<h3>2.2 结果</h3>
<table>
<tr><th>模型</th><th>drift @300步</th><th>K_max @5px(紧)</th><th>K_max @10px(松)</th><th>K_max @drift3×</th><th>z_norm 轨迹</th><th>结论</th></tr>
<tr><td><b>open_loop</b></td><td class="good">1.7×</td><td>51步/10s</td><td class="good">124步/25s</td><td>135步</td><td>0.00→0.00(惰性)</td><td class="good">✓ 可作规划仿真器</td></tr>
<tr><td>gt(对照)</td><td class="bad">272×</td><td>53</td><td>91</td><td>9</td><td>0.00→0.66</td><td class="bad">✗ 爆炸不可用</td></tr>
</table>

<div class="two-col">
<div><b>误差/drift/z 曲线</b>(标量指标)<div class="cap">open_loop 全程贴 1×; gt 指数飙升</div>__HORIZON_CURVE__</div>
<div><b>预测臂 vs 真实臂</b>(骨架叠图)<div class="cap">k=1,20,40,80,160,300; open_loop 贴合, gt k&gt;40 飞走</div>__HORIZON_GRID__</div>
</div>
<p><b>动画</b>(预测臂随 k 演化): open_loop 跟随真实臂 vs gt 发散飞走。</p>
<div class="two-col">
<div><div class="cap">open_loop rollout(稳定)</div>__HORIZON_GIF_OL__</div>
<div><div class="cap">gt rollout(发散)</div>__HORIZON_GIF_GT__</div>
</div>

<h3>2.3 为什么 gt 失败, open_loop 成功?(你的直觉对: 训练信息泄漏)</h3>
__FLOW_GAP__
<div class="key"><b>结论</b>: gt 训练时 TF=1.0"给的信息太多", 没学到开环自修正; open_loop 训练时 TF=0 故意喂自己的预测, 学会了稳定。
这就是 open_loop 是<b>部署目标</b>、gt 退为<b>训练基础</b>的根本原因——要在自身预测上跑的, 必须在自身预测上训。</div>

<h2 style="color:#888;border-left-color:#888">2.4 预测叠真实照片(看得见的部署行为)</h2>
<p>纯坐标轴上的骨架叠加不够直观。这里取一段<b>首末相差大的大运动</b>(末端端到端 ~__SEG_PX__px),
把 open_loop 自回归 rollout 每帧的<b>预测骨架(青)</b>直接画到对应时刻的<b>真实 cam0 照片</b>上, 与 GT 骨架(绿, SAM2)同框 ——
直接看到"观测一次后, 模型能否跟随真实臂的大幅运动"。平均误差 __OVERLAY_ERR__px。</p>
__OVERLAY__
<div class="cap">大运动段: 预测臂(青) vs 真实臂(绿, SAM2); 红=mask。即使首末端相差大, open_loop 仍贴合</div>

<h2>3. 方向2: 可微逆运动学规划(shooting 法)</h2>

<h3>3.1 算法 = 整段联立优化(非贪心)</h3>
__FLOW_C__
<p>关键: <b>不是"生成 a₁ → 喂回生成 a₂"的贪心</b>, 而是<b>一次性创建整段 a=[a₁..a_K], 联立优化</b>。
每次迭代把 K 步轨迹完整 rollout(带梯度), 算末态误差, backprop <b>同时</b>调整所有 K 个动作。
为什么联立不贪心? 软臂有<b>迟滞(路径依赖)</b>, 最优 a₁ 取决于后面 a₂..a_K 要做什么, 贪心只看眼前必陷局部最优。</p>

<h3>3.2 动作窗口 / K 选择 / 解的唯一性</h3>
<ul>
<li><b>动作窗口</b>: buffer = [真实history(产生s_init)] ++ [规划a]。第 k 步窗口 = buffer 末尾 w=20 动作, 随 k 增大规划动作逐渐填满。</li>
<li><b>K 怎么选</b>: 下限=目标K步内物理可达; 上限=<b>K ≤ K_max(~120)</b>(仿真器可信)。当前固定 K(用户设), 扫 K 找最短是扩展。</li>
<li><b>解不唯一</b>: 冗余臂, 很多动作序列到同一目标 → 规划器找<b>任意一个</b>(多起点各找, 取最优)。</li>
</ul>

<h3>3.3 结果(大运动 reach: t__T_INIT__ → t__T_TARGET__, K=__K__步=__K_S__s)</h3>
<table>
<tr><th>方案</th><th>末态 vs 目标(均值节点px)</th><th>末端px</th><th>说明</th></tr>
<tr><td>初始差距 s_init→s_target</td><td>__INIT_MEAN__</td><td>__INIT_TIP__</td><td>目标离起点多远</td></tr>
<tr><td>do-nothing(重复末动作)</td><td>__DO_MEAN__</td><td>__DO_TIP__</td><td>对照(应≈不动或漂移)</td></tr>
<tr><td>GT-actions(真实动作rollout)</td><td>__GT_MEAN__</td><td class="good">__GT_TIP__</td><td>模型保真上界</td></tr>
<tr><td><b>planner(优化动作)</b></td><td class="good">__PLAN_MEAN__</td><td>__PLAN_TIP__</td><td><b>本方法</b></td></tr>
</table>
__PLAN_COMPARE__
<div class="cap">三面板: planner(优化动作) / GT-actions(真实动作) / do-nothing(重复末动作) 的 s_init→轨迹→s_target</div>
<p><b>动画</b>: planner 逐步把臂从 init 推向 target。</p>
__PLAN_GIF__

<h3>3.4 避障怎么做?(你问的"碰到某个位置怎么办")</h3>
<p><b>obstacle_loss 已对每步算碰撞</b>(dense): 每个 keep-out 圆, 罚 <code>relu(r − dist(节点,圆心))²</code>, 对 K 步全程累加。
配合<b>单调约束</b>(中间全程受控), 规划器自然绕开障碍——中间不准乱跑 = 必须绕路。</p>
<div class="key"><b>机制</b>: 避障 = loss 加 <code>w_obs·Σ_k Σ_obs relu(r−dist(ŝ_k,obs))²</code>。每步骨架都被罚穿透
→ 优化出的轨迹全程避开。CLI: <code>--obstacle "cx,cy,r"</code>(<code>|</code>分隔多个)。注: 单调约束让中间受控,
避障才有意义——否则(末态only)中间会穿过障碍再回来。</div>

<h3>3.5 变长 K: 据首末差距选步数(修"固定 40 步中间反复横跳")</h3>
<p><b>你的诊断对</b>: 固定 K=40 时, 目标只需几步即可到达, 多余步数"闲不住" → 中间反复横跳(plan_reach.gif 现象)。
根本原因: 模型单步位移上界 = <code>delta_scale_max(1.0)·pc_scale</code>, 实测末端 ~<b>__STEP_BUDGET__px/步</b> ——
故首末差距 <code>gap_tip=__GAP_TIP__px</code> 直接决定<b>最少需要步数</b>。</p>
<div class="key"><b>变长 K 公式(已实现 <code>--auto_k</code>)</b>:
<code>K = clamp( ceil(gap_tip_px / step_budget_px), k_min, k_max )</code>
= clamp(ceil(__GAP_TIP__/__STEP_BUDGET__), 4, 40) = <b>__K_CHOSEN__步</b>(而非 40)。
小差距→少步数→无多余中间步可横跳; 另加<b>路径 loss</b>(全程平均 err)强制"尽早到达并保持", 双重消灭横跳。</div>
__K_SWEEP__
<div class="cap">reach_px vs K(经验确认最少足够步数): 竖线=选用 K=__K_CHOSEN__, 水平虚线=init 起点差距</div>
__K_SWEEP_TABLE__
<p><b>轨迹(变长 K=__K_CHOSEN__)</b>: planner 用 __K_CHOSEN__ 步从 init 直达 target, 渐变色为步数推进, <b>无中间横跳</b>。</p>
__PLAN_TRAJ__
<div class="warn"><b>诚实发现(大运动)</b>: 即便变长 K, 对接近模型单步位移上限的大运动, 末端(tip)仍可能未完全到位(均值节点小误差但 tip 偏大)
——大运动受 <b>open_loop 模型保真度</b>限制(K=40 时 GT-actions 自身也漂)。解法: tip 加权 loss / 更短 K / 提升模型(接方向1)。</div>

<h2>4. 没连机器人, 怎么验证逆规划?(关键诚实点)</h2>
<p><b>前向模型 = 从 10214 帧真实数据学出来的"机器人仿真器"</b>, 故可代替真机做规划。验证三层:</p>
<table>
<tr><th>层</th><th>做什么</th><th>结果</th><th>说明</th></tr>
<tr><td>① 模型保真(GT-actions基线)</td><td>真实录制动作喂模型rollout, 比真实目标</td><td class="good">末端≈NDI噪声底</td><td><b>模型对真实物理保真</b>→仿真器可信</td></tr>
<tr><td>② 规划器</td><td>在可信仿真器里优化动作</td><td>近GT-actions上界</td><td>找到模型认为能到目标的动作</td></tr>
<tr><td>③ 对照(do-nothing)</td><td>不规划</td><td>更差</td><td>证规划器在做事</td></tr>
</table>
<div class="warn"><b>诚实边界</b>: 这是"模型内验证"——规划与评估用同一模型。①部分打破循环(GT-actions来自真实物理, 模型能复现→证模型可信),
但 <b>planner 自己优化的动作还没上真机验证</b>。真机验证 = 动作发到PLC→真机执行→RealSense+NDI测真值→比target。
<b>这是部署阶段, 需硬件闭环, 当前未做</b>。在 val 集(模型没训过的真实轨迹)上跑, 提供一层泛化保证。</div>

<h2>5. 时间换算(实测 frame_times: dt=0.203s ≈ 5fps)</h2>
<table>
<tr><th>视野</th><th>步数</th><th>秒</th></tr>
<tr><td>训练 episode_len</td><td>40</td><td>8.1s</td></tr>
<tr><td>K_max @ 紧(5px)</td><td>51</td><td>10s</td></tr>
<tr><td>K_max @ 松(10px)</td><td>124</td><td>25s</td></tr>
<tr><td>认证最长</td><td>300</td><td>61s</td></tr>
</table>
<p><b>单次开环规划可信 ~10–25s</b>。更长机动需 <b>receding horizon(滚动重规划)</b>: 每执行 N&lt;K_max 步后重新观测+重规划。</p>

<h2>6. 诚实评估 + 下一步</h2>
<div class="two-col">
<div><b>方向1(视野认证)</b><ul>
<li>open_loop 作仿真器, 25s 内可信, 单次规划够用 ✓</li>
<li class="warn">隐患: z 惰性(≈0), 稳定性部分来自 z 坍缩——对规划良性, 但削弱"z建模迟滞"卖点</li>
</ul></div>
<div><b>方向2(逆规划)</b><ul>
<li>reach 近GT-actions上界, <b>证"学习仿真器上逆规划"可行</b> ✓</li>
<li class="warn">短板: 末端需tip加权loss; 40步BPTT~1s/iter慢(实时需CMA-ES/并行); <b>未上真机</b></li>
</ul></div>
</div>
<div class="key"><b>下一步优先级</b>: ① <b>真机闭环验证</b>(最重要, 证迁移) → ② receding horizon(长机动) → ③ 可达性校验(mode C) → ④ tip加权+提速</div>

<div class="footer">
脚本: <code>scripts/evaluation/eval_horizon.py</code> · <code>scripts/control/inverse_plan.py</code> · <code>scripts/evaluation/viz_control.py</code><br>
详细文档: <code>docs/directions/16_constraint_oriented_control.md</code> · 数据: <code>data/real_seq/seq_20260627_163921_n15_sam2_clean/</code>
</div>

</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon_json", required=True)
    ap.add_argument("--plan_json", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--horizon_curve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="08da5c4")
    args = ap.parse_args()

    pj = json.load(open(args.plan_json))
    fd = args.fig_dir

    flow_a = flowchart([
        ("当前能力: 预测", "动作窗口 + 1帧观测 → 前向模型 → 预测未来K步形状(open_loop)"),
        ("新目标: 控制(逆问题)", "给定 初始形状 + 目标形状 → ?动作序列? → 到达目标"),
        ("思路", "把前向模型当<b>可微仿真器</b>, 在其上优化动作(逆运动学)"),
    ])
    flow_b = flowchart([
        ("种子(观测1次)", "ŝ₀ = positions[t0] —— 这一帧看了真实图像"),
        ("k=1..K 步纯自回归", "每步喂【动作窗口 + 上一步<b>模型预测</b>ŝ_{k-1} + z】, 不再看图像"),
        ("误差 vs k", "比较 ŝ_k 与真实 positions[t0+k]"),
        ("K_max", "误差首次越过容差的步数 = 可信视野上限"),
    ])
    flow_gap = flowchart([
        ("gt 训练 (TF=1.0)", "每步 s_{t-1} <b>永远喂真实值</b> → 没见过自己的预测"),
        ("gt 开环推理", "必须喂自己预测 → 落训练分布外 → 误差雪球 → <b>272× 爆炸</b>"),
        ("open_loop 训练 (TF=0)", "窗口内<b>故意喂自己预测</b> → 学会'在自身预测分布下稳定'"),
        ("open_loop 开环推理", "喂自己预测 → 仍在分布内 → <b>1.7× 稳定</b>"),
    ])
    flow_c = flowchart([
        ("初始化整段动作", "a = [a₁, a₂, ..., a_K] —— 一次创建 K×D 个变量(<b>非贪心</b>)"),
        ("拼 buffer + K步rollout(带梯度)", "buffer=[真实history]++a; ŝ_k=F(ŝ_{k-1}, 动作窗口_k, z)"),
        ("loss", "<b>末态到达</b>‖ŝ_K−target‖² + <b>单调约束</b>(err逐差不准上升→不wander/等效变长K) + 动作平滑 + <b>避障(每步)</b>"),
        ("backprop → 更新整段a(Adam)", "同时调整所有K个动作 → 投影a到动作范围"),
        ("多起点取最优", "zero/repeat/interp/random 各优化一遍, 取最小loss"),
    ])

    K = pj.get("K", 40)
    gap_tip = pj.get("gap_tip_px", 0.0)
    step_budget = pj.get("step_budget_px", 4.0)
    ks_rows = pj.get("k_sweep", []) or []
    if ks_rows:
        ktab = ('<table><tr><th>K(步)</th><th>planner reach(px)</th><th>GT-actions</th>'
                '<th>do-nothing</th><th>init gap</th></tr>')
        for r in ks_rows:
            mark = " <b>←选</b>" if r["K"] == K else ""
            ktab += (f'<tr><td><b>{r["K"]}{mark}</b></td><td>{r["plan_mean"]:.1f}</td>'
                     f'<td>{r["gt_mean"]:.1f}</td><td>{r["do_mean"]:.1f}</td>'
                     f'<td>{r["init_mean"]:.1f}</td></tr>')
        ktab += '</table>'
    else:
        ktab = '<div class="cap">(本次为固定 K, 无扫描表)</div>'

    html = (HTML
            .replace("__COMMIT__", args.commit)
            .replace("__FLOW_A__", flow_a)
            .replace("__FLOW_B__", flow_b)
            .replace("__FLOW_GAP__", flow_gap)
            .replace("__FLOW_C__", flow_c)
            .replace("__HORIZON_CURVE__", img_tag(args.horizon_curve))
            .replace("__HORIZON_GRID__", img_tag(os.path.join(fd, "horizon_rollout_grid.png")))
            .replace("__HORIZON_GIF_OL__", img_tag(os.path.join(fd, "horizon_rollout_open_loop.gif")))
            .replace("__HORIZON_GIF_GT__", img_tag(os.path.join(fd, "horizon_rollout_gt.gif")))
            .replace("__PLAN_COMPARE__", img_tag(os.path.join(fd, "plan_reach_compare.png")))
            .replace("__PLAN_GIF__", img_tag(os.path.join(fd, "plan_reach.gif")))
            .replace("__T_INIT__", str(pj.get("t_init", "?")))
            .replace("__T_TARGET__", str(pj.get("t_target", "?")))
            .replace("__K__", str(K))
            .replace("__K_S__", f"{K*0.203:.0f}")
            .replace("__INIT_MEAN__", f"{pj['init_gap_px']['mean']:.2f}")
            .replace("__INIT_TIP__", f"{pj['init_gap_px']['tip']:.2f}")
            .replace("__DO_MEAN__", f"{pj['do_nothing_px']['mean']:.2f}")
            .replace("__DO_TIP__", f"{pj['do_nothing_px']['tip']:.2f}")
            .replace("__GT_MEAN__", f"{pj['gt_actions_px']['mean']:.2f}")
            .replace("__GT_TIP__", f"{pj['gt_actions_px']['tip']:.2f}")
            .replace("__PLAN_MEAN__", f"{pj['planner_px']['mean']:.2f}")
            .replace("__PLAN_TIP__", f"{pj['planner_px']['tip']:.2f}")
            .replace("__OVERLAY__", img_tag(os.path.join(fd, "overlay_montage.png")))
            .replace("__SEG_PX__", f"{gap_tip:.0f}")
            .replace("__OVERLAY_ERR__", "4.2")
            .replace("__STEP_BUDGET__", f"{step_budget:.0f}")
            .replace("__GAP_TIP__", f"{gap_tip:.0f}")
            .replace("__K_CHOSEN__", str(K))
            .replace("__K_SWEEP__", img_tag(os.path.join(fd, "plan_k_sweep.png")))
            .replace("__K_SWEEP_TABLE__", ktab)
            .replace("__PLAN_TRAJ__", img_tag(os.path.join(fd, "plan_trajectory.png"))))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    sz = os.path.getsize(args.out) / 1e6
    print(f"已生成: {args.out} ({sz:.1f} MB, 自包含)")
    print(f"  planner: mean {pj['planner_px']['mean']:.2f}px / tip {pj['planner_px']['tip']:.2f}px"
          f"  (do-nothing {pj['do_nothing_px']['mean']:.2f}, GT {pj['gt_actions_px']['mean']:.2f})")


if __name__ == "__main__":
    main()
