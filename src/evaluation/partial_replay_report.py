"""Standalone visual replay report with frame slider, pressure and forecast plots."""
from __future__ import annotations

import base64
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_report(output,summary,audit,metrics,steps,traces,panels):
    scale=traces["action_unit_to_kpa"]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for key,label in (("factual_open_mm","No correction"),("factual_prior_mm","Before this update"),
                      ("factual_corrected_mm","After this update")):
        axes[0,0].plot([r[key] for r in metrics],label=label)
    axes[0,0].set(title="Recorded-input prediction vs pseudo-reference",ylabel="Mean node error (mm)")
    axes[0,0].legend(fontsize=8)
    for key,label in (("factual_hidden_prior_mm","Hidden before"),("factual_hidden_corrected_mm","Hidden after")):
        axes[0,1].plot([r[key] for r in metrics],label=label)
    axes[0,1].set(title="Nodes inside the fixed square",ylabel="Mean node error (mm)")
    axes[0,1].legend(fontsize=8)
    for c in range(traces["assumed_actions"].shape[1]):
        color = plt.get_cmap("tab10")(c)
        axes[1,0].plot(traces["assumed_actions"][:,c]*scale[c],label=f"assumed ch{c}",color=color)
        axes[1,0].plot(traces["recorded_actions"][:,c]*scale[c],linestyle="--",alpha=.4,color=color)
    axes[1,0].set(title="Assumed issued (solid) / recorded (dashed)",ylabel="Pressure (kPa)")
    axes[1,0].legend(fontsize=7,ncol=2)
    axes[1,1].plot([r["suffix_ms"] for r in metrics],label="Full suffix correction B")
    axes[1,1].plot([r["state_ms"] for r in metrics],label="State correction (excludes detector)")
    axes[1,1].set(title="CPU offline computation, no hardware deadline",ylabel="Time (ms)")
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Replay step");ax.grid(alpha=.2)
    fig.savefig(output/"overview.png",dpi=150);plt.close(fig)
    # JSON has no NaNs: padded future tensors are serialized only over valid suffixes.
    frames=[]
    for k,panel in enumerate(panels):
        frame={"image":"data:image/jpeg;base64,"+base64.b64encode(panel).decode(),
               "metrics":metrics[k],"details":steps[k],
               "remaining_before_kpa":(traces["suffix_before"][k,k+1:]*scale).tolist(),
               "remaining_after_kpa":(traces["suffix_after"][k,k+1:]*scale).tolist(),
               "future_terminal_before":traces["future_before_state"][k,-1].tolist() if k<len(panels)-1 else None,
               "future_terminal_after_state":traces["future_after_state"][k,-1].tolist() if k<len(panels)-1 else None,
               "future_terminal_after_control":traces["future_after_control"][k,-1].tolist() if k<len(panels)-1 else None}
        frames.append(frame)
    payload={"frames":frames,"summary":summary,"audit":audit,"target_terminal":traces["reference"][-1].tolist(),
             "initial":(traces["initial_plan"]*scale).tolist(),"recorded":(traces["recorded_actions"]*scale).tolist(),
             "assumed":(traces["assumed_actions"]*scale).tolist()}
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>部分观测逐帧回放验证</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:24px auto;max-width:1280px;padding:0 20px;color:#1a293b;background:#f5f7fa}
h1{font-size:27px} .card{background:white;border:1px solid #d9e1eb;border-radius:10px;padding:18px;margin:16px 0}
img{width:100%;border-radius:6px}canvas{width:100%;height:300px} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
input[type=range]{width:70%}button{padding:6px 18px;margin-right:12px;cursor:pointer}pre{white-space:pre-wrap;font-size:12px;max-height:420px;overflow:auto}
.muted{color:#53667d}#status{font-weight:600} @media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><h1>固定遮挡下的逐帧状态与压力修订</h1>
<p>左图为原始采集，右图为遮挡图像。橙色为本帧预测，绿色为历史校正后骨架，青点为图像边缘候选。edges 模式以这些边缘校正；nodes 模式使用已知可见参考节点，仅作理想测量对照。</p>
<div class="card"><strong>实验含义</strong><p>主回放假设执行新计划，每步注入原录制图像，用于检查反馈计算顺序。新动作没有产生这些录制图像；下面的压力变化及模型内目标改善不能解释为实机闭环结果。预测精度另由始终使用原采集动作的对照流评价。</p><p id="setup" class="muted"></p></div>
<div class="card"><button id="play">播放</button><input id="slider" type="range" min="0" value="0"><span id="index"></span><p id="status"></p><img id="frame"></div>
<div class="grid"><div class="card"><strong>本步修订前后：整个剩余压力序列</strong><p class="muted">实线：新后缀；虚线：旧后缀。颜色对应 4 个有效通道，单位 kPa。</p><canvas id="pressure" width="600" height="300"></canvas></div>
<div class="card"><strong>终点形状预测如何变化</strong><p class="muted">灰：目标；橙：状态校正前；蓝：状态校正后、动作未改；绿：动作也修订后。单位 mm。</p><canvas id="shape" width="600" height="300"></canvas></div></div>
<div class="card"><strong>整段对照</strong><img src="data:image/png;base64,__OVERVIEW__"></div>
<details class="card"><summary>当前帧状态校正、动作求解与输入差异</summary><pre id="details"></pre></details>
<details class="card"><summary>结果与数据配对边界</summary><pre id="summary"></pre></details>
<script>const DATA=__PAYLOAD__;
const slider=document.querySelector('#slider');slider.max=DATA.frames.length-1;
document.querySelector('#setup').textContent=`观测模式：${DATA.summary.measurement}；原始帧 ${DATA.audit.raw_frame_slice.join(' 至 ')}（右端不含）；形状变化指标 ${DATA.audit.selected_motion_spread_mm.toFixed(2)} mm；固定方块 xywh=${DATA.audit.occluder_xywh.join(', ')}。`;
document.querySelector('#summary').textContent=JSON.stringify({summary:DATA.summary,audit:DATA.audit},null,2);
const colors=['#d46a24','#2377b8','#299d70','#9c4daf'];
function plot(id,series,invertY=false){const c=document.getElementById(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
let points=series.flatMap(s=>s.p);if(!points.length){ctx.fillText('序列结束，无剩余动作',30,40);return;}
let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
if(id==='shape'){let aspect=(c.width-65)/(c.height-45),span=Math.max((xmax-xmin)/aspect,ymax-ymin,1),xc=(xmin+xmax)/2,yc=(ymin+ymax)/2;xmin=xc-span*aspect/2;xmax=xc+span*aspect/2;ymin=yc-span/2;ymax=yc+span/2;}
let xr=Math.max(xmax-xmin,1),yr=Math.max(ymax-ymin,1),left=45,top=15,w=c.width-65,h=c.height-45;
ctx.strokeStyle='#dce3eb';ctx.strokeRect(left,top,w,h);ctx.fillStyle='#52657a';ctx.font='12px sans-serif';
ctx.fillText((invertY?ymin:ymax).toFixed(1),2,top+10);ctx.fillText((invertY?ymax:ymin).toFixed(1),2,top+h);ctx.fillText(xmin.toFixed(0),left,top+h+20);ctx.fillText(xmax.toFixed(0),left+w-20,top+h+20);
for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.setLineDash(s.dash?[5,4]:[]);ctx.beginPath();s.p.forEach((p,i)=>{let x=left+(p[0]-xmin)/xr*w;let y=top+(invertY?(p[1]-ymin)/yr:1-(p[1]-ymin)/yr)*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}ctx.setLineDash([]);}
function show(){let k=+slider.value,f=DATA.frames[k],m=f.metrics;document.querySelector('#index').textContent=` ${k+1} / ${DATA.frames.length}`;
document.querySelector('#frame').src=f.image;document.querySelector('#status').textContent=`原始帧 ${m.raw_frame} · 边缘 ${m.edge_count} · 状态校正 ${m.state_accepted?'接受':'未接受'} · 后缀修订 ${m.suffix_accepted?'接受':'未接受'} · 假设/录制压力最大差 ${m.action_mismatch_kpa.toFixed(2)} kPa`;
document.querySelector('#details').textContent=JSON.stringify(f.details,null,2);let ps=[];
for(let c=0;c<4;c++){ps.push({p:f.remaining_before_kpa.map((v,i)=>[k+1+i,v[c]]),color:colors[c],dash:true});ps.push({p:f.remaining_after_kpa.map((v,i)=>[k+1+i,v[c]]),color:colors[c]})}plot('pressure',ps);
let ss=[{p:DATA.target_terminal,color:'#87919e'}];for(const [key,color] of [['future_terminal_before','#d46a24'],['future_terminal_after_state','#2377b8'],['future_terminal_after_control','#299d70']])if(f[key])ss.push({p:f[key],color});plot('shape',ss,true);}
slider.addEventListener('input',show);let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放'}else{document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{slider.value=(+slider.value+1)%DATA.frames.length;show()},200)}};show();</script></html>'''
    encoded=base64.b64encode((output/"overview.png").read_bytes()).decode()
    html=template.replace('__OVERVIEW__',encoded).replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (output/"report.html").write_text(html,encoding="utf-8")
