"""Self-contained operating guide from screenshots of the exercised Qt GUI."""
import base64
import html
import json
from pathlib import Path


def write_guide(out, captures, summary):
    out = Path(out)
    nav = ''.join(f'<a href="#{c["name"]}">{html.escape(c["title"])}</a>' for c in captures)
    sections = []
    for capture in captures:
        encoded = base64.b64encode((out/(capture['name']+'.png')).read_bytes()).decode()
        labels = ''.join(f'<li>{html.escape(label)}</li>' for label in capture['labels'])
        sections.append(f'<section id="{capture["name"]}"><h2>{html.escape(capture["title"])}</h2>'
                        f'<p>{html.escape(capture["description"])}</p><ol>{labels}</ol>'
                        f'<a href="#" class="zoom" aria-label="放大截图"><img alt="{html.escape(capture["title"])}，带编号按钮标注" '
                        f'src="data:image/png;base64,{encoded}"></a></section>')
    p50, p95, maximum = summary['compute_ms_percentiles']
    compute_text=('本次对照模式没有运行矫正，不报告矫正计算时延。' if p50 is None else f'反馈计算 P50 / P95 / max：{p50:.1f} / {p95:.1f} / {maximum:.1f} ms。')
    intervals=summary.get('command_interval_ms_percentiles')
    interval_text='' if intervals is None else '实际发令间隔 P50 / P95 / max：'+' / '.join(f'{v:.1f}' for v in intervals)+' ms；已包含本次虚拟阀、等图、反馈计算与保存引起的周期延迟。'
    initial=summary.get('initial_planning',{})
    deadline_text=(f'本次初始规划 {initial.get("ms",0):.1f} ms；反馈提交/跳过计数：{html.escape(json.dumps(summary.get("revision_status_counts",{}),ensure_ascii=False))}。')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>实机验证工作台 · 实验操作手册</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#183444;font:16px/1.8 system-ui,"Noto Sans CJK SC",sans-serif}
header{background:#143d48;color:white;padding:44px max(24px,calc((100% - 1180px)/2))}header p{color:#d8e8e8}main{max-width:1180px;margin:auto;padding:24px}
nav{display:flex;flex-wrap:wrap;gap:8px}nav a{background:white;border:1px solid #cfdddd;border-radius:8px;padding:6px 12px;color:#075762;text-decoration:none}
section,.card{background:white;border:1px solid #dde6e8;border-radius:12px;padding:24px;margin:24px 0;scroll-margin-top:12px}
h1{line-height:1.3;font-size:34px}h2{line-height:1.4}img{width:100%;height:auto;border:1px solid #d5dfe4;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #d5dfe4;text-align:left}
code{background:#edf3f5;padding:3px 6px;overflow-wrap:anywhere}.notice{border-left:4px solid #d28c26;padding-left:16px}dialog{max-width:98vw;max-height:98vh;padding:8px;border:0}dialog img{width:auto;max-width:94vw;max-height:88vh}button{font:inherit;cursor:pointer}a:focus-visible,button:focus-visible{outline:3px solid #cc7200}footer{padding:24px;color:#516976}@media print{nav,dialog{display:none}section{break-inside:avoid}header{background:white;color:black}}</style></head><body>
<header><p>SelfSoftRobot · 2026-09-11 · Analytic B</p><h1>连接、部署预热、规划预览与反馈执行</h1><p>四个顺序页面，一个全局六腔调压窗口。实际 Qt 窗口截图，红圈数字对应每节操作；点击截图放大。</p></header><main>
<div class="card"><h2>先看阶段的区别</h2><table><tr><th>阶段</th><th>动作与状态</th><th>结束条件</th></tr>
<tr><td>实验前准备</td><td>加载模型、紧凑六腔映射、分别连接设备。加载不发压力，也不记录准备动作；可用独立六腔工具手动调到初始压力。</td><td>设备连接完成，结束手动调压并保持。</td></tr>
<tr><td>初始位置保持</td><td>提取按钮确认当前保持压力并建立部署历史；检查或修改黄点。确认后固定相机变换，估计 p/h，保持预热。</td><td>新图像质量、残差与稳定性连续达标，显示模型就绪。</td></tr>
<tr><td>正式计算与控制</td><td>按平均与最大节点容限自动搜索长度；播放模型轨迹并检查六腔压力。执行才发送动作、校正状态及修订剩余动作。</td><td>完成或普通停止保持末压并继续观测；归零是独立动作。手动接管先停止自动发送，旧计划作废。</td></tr></table>
<p>默认启动：<code>python -m real_validation.main</code>。旧 OpenLoop 的五页流程移到显式兼容入口 <code>--legacy-openloop</code>，不再作为本实验的前五个必经步骤。</p>
<p>服务器上先用全 Mock；实机 USB / COM 连接留在原接口。独立目录安装 <code>requirements.txt</code>，连接真设备按需安装 <code>requirements-hardware.txt</code>；SAM初始化另装 <code>requirements-sam2.txt</code>。建议Python3.10（64位）；基础依赖已含传统感知，安装方法见README。YOLO尚未接入，后续发行清单见 <code>YOLO_DEPLOYMENT_PLAN.md</code>。控制部署包是同名 <code>hereditary.npz</code> 与 <code>hereditary.json</code>。</p></div>
''' + f'<nav aria-label="操作步骤">{nav}</nav>' + ''.join(sections) + '''
<section><h2>本次全量训练权重与模型周期</h2><p>新版独立包默认加载全平台新 10 Hz 权重，旧参考权重保存在候选目录，之前的 ZIP 不覆盖。第1页“选择模型”可打开 <code>checkpoints/candidates/</code>，选择：</p><ul><li><code>hereditary_full_5hz_300ep_20260910_001.npz</code>：dt=0.2 s，训练300轮，验证集选择第150轮。</li><li><code>hereditary_full_10hz_300ep_20260910_001.npz</code>：dt=0.1 s，训练300轮，验证集选择第300轮。</li><li><code>hereditary_reference_10hz_20260909.npz</code>：此前工作台参考模型。</li></ul><p>每套NPZ必须与同名JSON一起保留。加载自动读取JSON中的dt，控制周期、限速换算及记忆状态推进均使用该值；不能只修改JSON来换频率，应加载对应模型。相机和界面刷新频率独立，实际发令间隔仍可能更长。</p><p>本文截图与统计来自虚拟实验，模型来源见运行记录，只演示操作与软件流程。独立包检查记录另存于 <code>validation_records/</code>；所有模拟结果均不是实机精度证明。腔道接线实拍另见包内 <code>CHAMBER_REFERENCE.html</code>。</p></section>
<section><h2>精简初始化与重复实验</h2><p>默认第二页只需自动提取、检查黄线、确认部署。SAM 自动生成图像候选提示，无需先框选；有歧义时在臂身中间点一下重试。框选、负提示、手绘、模型设备和预热阈值都在“修正形状 / 高级设置”中。首次加载仍可能数秒；日志分别记录加载、编码、解码时间。</p><p>绿色是冻结帧分割，青色是模型估计，白点是当前可见边缘；遮挡区域没有白点，不表示测得完整形状。手动调压时青色按ACK历史预测，保持后恢复图像修正。</p><p>清零或手动调压后，配准有效、历史已知时，同一确认按钮改为“保持当前压力，重新预热（复用配准）”；不重新分割，也不由预热发送气压。相机改变或历史未知才需重新恢复相应条件。六腔工具显示真实生效的升降速率，目前候选模型部署上限50 kPa/s。</p></section><section><h2>执行对照：关闭矫正</h2><p>第四页勾选“不使用矫正，仅执行规划（对照实验）”，开始按钮会同步显示“不矫正”。执行中不调用图像状态校正、不优化剩余动作；原图、软件遮挡图、压力、NDI和时延继续记录。记录中的control_mode为open_loop，correction_enabled为false，逐步revision_status为disabled，不会把未运行的矫正写成0毫秒测量。</p><p>默认不勾选，使用Analytic B。切换模式撤销执行确认但保留规划，执行中锁定。预热和执行外的保持观测仍可估计状态；对照关闭的是正式运动阶段的矫正。ACK对应压力若与请求不同，仍按相同压力/限速规则修正接续并记录；无新图像仍受采集失败停止门限约束，全遮挡的有效图像则只保存，不因缺乏边缘停止。比较时应尽量复现相同初始形状、持压历史、目标和扰动，并记录是否复用同一计划；两个连续试次本身不能保证初始记忆相同。</p></section><section><h2>规划参数与在线期限</h2><p>第三页「初始规划参数…」可调整容限、最大长度、末端调整余量、总预算、B 迭代次数、终态优化评估次数及长度搜索步长。余量默认10步，0表示额外0步，不是0kPa、不会清零；已计入预览与执行总长，开环和闭环使用相同时长。末端以保持压力起步供在线B修订，用尽后保持，记录估计残差，不无限重规划。修改立即生效并使旧计划失效。总预算在迭代间检查，单次调用可能略超；不是固定执行步数。</p><p>反馈在独立快照中运行，只有本次发令时间 + 模型 dt − 3 ms 前完成才一起提交状态和后缀。迟到结果整次丢弃；后台只容纳一个任务，不积压。连续跳过默认 10 次停止归零，可在第四页调整。额外等图默认 0 ms，始终要求 ACK 后新图；调大等待会减少计算预算。</p><p>timings.csv 保存逐步阶段耗时，steps.jsonl 保存修订前/ACK 后合法后缀/最终采用序列及原因，initial_plan.npz 保存原计划；feedback_jobs 保存候选及完整实际计算耗时，迟到候选不代表已采用。实时显示的是等待时间，尚未完成的计算不记作 0。图片后台有界保存，samples.csv 另列写图时间。线程/GIL、相机与串口仍有抖动，这是软期限，不保证严格 100 ms。</p></section>
<section><h2>规划与到位的判据</h2><p>完整形状与局部目标互斥，切换清除旧目标。局部可单击末端点，指定节点的一段，或选择“任意臂段自动匹配”后只画一段、不指定节点。自动模式在总预算内搜索连续节点区间与两个绘制方向，用32点覆盖完整目标曲线；预览标注选中区段，执行中固定对应关系。无需从 base 开始；方向为近 base 端到近 tip 端。只拟合选中节点，其余由模型规划。默认受约束节点平均容限 2 mm、最大节点 4 mm；可以自行调整。搜索上限是计算保护，不是必须执行的长度。外层依次尝试更长时域，复用候选，并加入限速的终态压力搜索；在线仍用 Analytic B。未达标仍保留当前合法计划；检查预览后可在第4页勾选“允许试运行未达标计划”，再确认执行。计划仍标记未达标，日志保存误差与试运行标记；压力、状态一致性及反馈停止条件仍生效。</p><p>拖动时间轴查看紫色预测轨迹，青色是当前模型估计，红色是目标。压力曲线与六腔数值属于该预览。执行前从当前保持状态重验原计划；预览因状态变化而失效时要求重新预览，不暗中执行另一条轨迹。</p><p>预热默认最短 2 s、最多 20 s，连续 8 个新帧，侧边覆盖至少 75%、残差不超过 3 px、帧间估计形状变化不超过 1 px；这些是可调工程门限，不是物理记忆已经唯一识别的认证。初始化应无遮挡。保持阶段固定相机变换，按实际时间推进记忆；正式结束后继续显示模型目标差和可信可见边缘目标差，遮挡部分不计作实测全形状。</p></section>
<section><h2>A 与 Analytic B：结果究竟差多少</h2>
<p>同一高变化窗口，序列 182519、原始帧 2198–2277，共 80 帧，固定 56×56 方块。下表形状均为图像校正后模型预测，与录制参考形状比较；排除固定 base，距离单位 mm。</p>
<table><tr><th>指标</th><th>Cached A</th><th>Analytic B</th></tr>
<tr><td>80 帧平均节点距离</td><td>0.7474</td><td>0.7064</td></tr>
<tr><td>最后一帧平均节点距离</td><td>0.5723</td><td>0.3668</td></tr>
<tr><td>最后一帧末端点距离</td><td>1.0696</td><td>0.1712</td></tr>
<tr><td>剩余动作修订接受</td><td>16 / 79</td><td>78 / 79</td></tr>
<tr><td>完整反馈计算 P50 / P95 / max（ms）</td><td>22.1 / 27.7 / 33.8</td><td>39.8 / 48.1 / 54.6</td></tr></table>
<p>A 的 7 次拒绝来自缓存失配，另 56 次是候选投影到压力/速率约束后，没有通过真实非线性 rollout 的下降校验；拒绝时保留原剩余计划。状态观察器仍在更新，所以接受次数并不是控制成功率。</p>
<p class="notice">本窗口 B 更好，但录制图像并不响应修改后的压力，因此不能据此宣称实机到位精度提高。上述计时不包含相机与阀通信；B 另一次独立进程的最大值为 139.2 ms，也不能宣称严格 100 ms 截止期。</p>
<p>数据来源：<code>workspace/runs/analysis/partial_observation_latency/cpu_20260909_001/{cached_a,fast_b}/traces.npz</code>；参考形状来自 <code>partial_observation_replay/edges_20260908_001/traces.npz</code>。最后一帧没有剩余动作，不计入 79 次修订。</p></section>
<section><h2>软件遮挡、全遮挡与可选设备</h2><p>第四页的软件遮挡可用于真实和虚拟相机。每行设置 x/y/宽/高百分比和灰度，最多 16 个矩形；先应用再执行。初始化前关闭。它只修改送给观察器的图像副本，矩形位置不进入边缘判断。物理遮挡不需要手工标注。</p><p>完全没有新图或可信边缘时默认最多连续 3 次，第四页可改为 1–10。缺失期间仅使用模型和合法后缀，到限后停止归零、撤销就绪，不无限搜索或试探压力。短计划结束时若仍缺少反馈，也撤销就绪并标记未确认，不能直接继续规划。全遮挡下没有稳定跟踪保证，有纹理的遮挡还可能造成边缘误匹配。</p><p>第一页可选连接 NDI（backend、COM 端口、探头数），不连接也能执行。相机驱动可选自动、RealSense SDK 或 UVC/OpenCV；自动优先枚举 SDK 设备，未发现时用配置的视频设备索引。明确选定 SDK/serial 之后不会在失败时悄悄换相机。仅采集彩色流，不保存深度图。</p><p>每次执行目录包含 commands.csv、raw/camN 原图、frames 反馈使用图、samples.csv、ndi.csv、frame_ndi.csv 和 metadata.json。NDI 只作评价，不输入控制；失锁保留 NaN。所有关联使用主机接收时间，辅助相机新鲜性与 NDI 年龄有明确记录，不称为硬件同步。故障归零和最后的失败图也保存。</p></section>
<section><h2>自动遮挡处理的准确含义</h2><p>反馈只沿预测臂身的侧向寻找图像边缘，用对比度、方向、颜色和候选歧义过滤，再用可信边缘到预测管身的距离校正 p/h。无须提供遮挡方块的位置，也不会穿过缺口重新拼成完整骨架。</p>
<p>界面显示的覆盖率，是可用侧边缘覆盖的内部线段比例，排除 base 附件与末端帽；不是可见像素比例，也不是遮挡概率。边缘缺失只能提示可能遮挡。当前没有通用遮挡物分割器；颜色相似的遮挡物仍可能被误关联。缺少边缘的区域不应被当作真实测量。</p>
<p>初始化推荐直接自动提取，SAM2.1 Tiny 使用图像候选提示；歧义时点一下臂身，修正工具中仍保留框选和正/负提示；近景不要求臂身只占画面一小块。也可直接手绘，无须先让自动提取成功。手绘微调默认保留人工 BASE/TIP；算法未识别短边不会阻止使用人工端点。初始化在后台联合拟合尺度、旋转和有界记忆，拟合后按臂长比例检验，再进行连续图像预热。SAM 仅在初始化使用，实时反馈仍为轻量边缘观察器。</p><p>独立包带分割权重和上游源码，需安装 requirements-sam2.txt；CPU 可运行，GPU 自动检测。SAM 掩膜、分数和预热通过均不替代人工检查完整形状；相似变换不能校正任意透视畸变。绿色掩膜、提示点、人工端点来源和参数拟合均保存到本次实验目录。</p></section>
''' + f'''<section><h2>本次虚拟设备检查</h2><p>实际 GUI 完成模型加载、部分阀组连接、独立手动调压、实时限制校验、非零压力部署预热、像素提取与编辑、自动时域搜索、轨迹播放、遮挡反馈、手动接管与归零。</p>
<p>{summary['commands']} 条执行 ACK、{summary['feedback_frames']} 帧反馈。{compute_text}归零 ACK 已确认。这不是相机到阀门的总延迟，也不是物理控制测量。</p>
<p>{interval_text}</p><p>{deadline_text}</p><p>原始运行：<code>{html.escape(summary['run_dir'])}</code>。截图的未标注原版保存在原始运行目录；独立包中的本HTML已内嵌操作截图，不依赖相邻图片目录。</p></section>''' + '''
</main><dialog id="viewer"><button type="button" id="close">关闭放大图</button><br><img alt="放大的操作截图"></dialog>
<footer>此文件内嵌全部截图，不需要网络或相邻图片即可打开；浏览器打印可导出 PDF。</footer>
<script>const d=document.querySelector('#viewer');document.querySelectorAll('.zoom').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();d.querySelector('img').src=a.querySelector('img').src;d.querySelector('img').alt=a.querySelector('img').alt;d.showModal()}));document.querySelector('#close').onclick=()=>d.close();d.addEventListener('click',e=>{if(e.target===d)d.close()});</script></body></html>'''
    (out/'guide.html').write_text(page, encoding='utf-8')
    (out/'captures.json').write_text(json.dumps(captures, ensure_ascii=False, indent=2)+'\n')
