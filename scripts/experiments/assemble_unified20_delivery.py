#!/usr/bin/env python3
"""Collect publication exports and verify the completed report's local assets."""
import argparse
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from plot_unified20_results import ROOT, OUT, FIG, REPORT, read, write


def collect():
    records={r['id']:r for r in read(OUT/'figure_catalog.json')}
    for name in ['figure_efficiency_sampling.json','figure_geometry.json']:
        path=OUT/name
        if path.exists():
            for r in read(path):records[r['id']]=r
    ordered=['prediction_summary','memory_ablation','geometry_mechanism','time_kernel_structure','time_kernel_gain','memory_plugin','learning_and_inference','sampling_alignment','sequence_and_capacity','shape_examples']
    figures=[records[x] for x in ordered if x in records]
    assert len(figures)==len(records),set(records)-set(ordered)
    for row in figures:
        for extension in ['svg','pdf','png']:
            for folder in [FIG,REPORT/'figures']:
                path=folder/f'{row["id"]}.{extension}'
                assert path.is_file() and path.stat().st_size>1000,path
        ET.parse(FIG/f'{row["id"]}.svg')
    write(OUT/'figure_catalog.json',figures)
    design=read(OUT/'figure_design.json');design['figures']=figures
    design['example_selection']='Maximum absolute target endpoint x per recording, seed 100; shape selection independent of model prediction error. Local-error panels share a scale.'
    write(OUT/'figure_design.json',design)
    lines=['# 统一20次重复实验：结果交付记录','',
      '训练来源：`workspace/runs/training/modeling_unified20_20260913_004`。分析来源：`workspace/runs/analysis/modeling_unified20_20260913_005`。',
      '', '主训练共286个拟合：14种随机配置各20次、6种确定性表示各一次。保留全部seed 100–119。主指标为2,958个测试目标帧的合并平均，各seed等权汇总。',
      '', '## 阅读入口','',
      '- [HTML结果与机制分析](report.html)',
      '- [原始canonical artifact](artifact.json)',
      '- 论文：`docs/icra2027/draft.md`；修改前快照在同目录的`archive/draft_before_unified20_results_20260913.md`。',
      '', '## 图目录','',
      '全部图同时导出SVG、PDF和250 dpi PNG。论文图采用蓝色、橙色和中性色，线型及符号辅助区分。标准差、配对置信区间和时延分位数分别标注，不能互相替代。','']
    for row in figures:
        lines.extend([f'### {row["id"]}','',f'[SVG](figures/{row["id"]}.svg) · [PDF](figures/{row["id"]}.pdf) · [PNG](figures/{row["id"]}.png)','',row['caption'],''])
    lines.extend(['## 重建图表和HTML','',
      '在项目根目录运行以下命令；仅读取本轮已有训练与分析结果，不重新训练。Python环境为`/Data5/ddf/environments/conda_envs/selfsr/bin/python`。','',
      '```bash',
      'python scripts/experiments/plot_unified20_results.py',
      'python scripts/experiments/plot_unified20_efficiency_sampling.py',
      'python scripts/experiments/plot_unified20_geometry.py',
      'python scripts/experiments/assemble_unified20_delivery.py',
      'python scripts/experiments/build_unified20_report.py --deliver',
      'python scripts/experiments/assemble_unified20_delivery.py --verify-report --bundle',
      '```','',
      '各专项分析的复算脚本为`analyze_unified20_geometry.py`、`analyze_unified20_efficiency.py`、`analyze_unified20_sampling.py`；统计独立核验脚本保存在analysis的`validation/`目录。',
      '', '## 解释范围','',
      '统计区间描述固定数据与划分下的优化随机性。100 epoch是统一预算，部分模型在末段仍有改善。数据子集和部分结构容量来自此前探索，新seed不构成全新采集测试。',
      '', '时间核刻画学习读出与局部几何的合成作用；低秩描述与两分支夹角不构成唯一物理机制识别。10 Hz补充记录使用旧视觉标签，stride-2比较不等于重新以5 Hz执行。CPU模型求值延迟不包含图像感知、通信、规划和执行。',
      '', '硬件待确认信息、独立尺度标定与本轮尚未完成的真实控制结果仍在论文中明确留空。',
      '', 'HTML通过canonical数据/结构构建与本地资源核验；当前环境未实测浏览器交互，浏览器验证范围以portable builder回执为准。'])
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    return figures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-report',action='store_true');parser.add_argument('--bundle',action='store_true');args=parser.parse_args()
    records=collect();checks=dict(figures=len(records),formats_per_figure=3,svg_xml='valid',figure_files='present',report_checked=False)
    if args.verify_report:
        html=(REPORT/'report.html').read_text();assert len(html)>10000
        artifact=read(REPORT/'artifact.json')
        assert artifact['manifest']['title'] in html
        class Links(HTMLParser):
            def __init__(self):super().__init__();self.hrefs=[]
            def handle_starttag(self,tag,attrs):
                if tag=='a':self.hrefs.extend(v for k,v in attrs if k=='href' and v)
        parsed=Links();parsed.feed(html)
        for r in records:
            for extension in ['svg','pdf','png']:
                target=f'figures/{r["id"]}.{extension}'
                assert any(h in (target,'./'+target) for h in parsed.hrefs),target
        draft=ROOT/'docs/icra2027/draft.md'
        for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',draft.read_text()):
            if not image.startswith(('http://','https://')):assert (draft.parent/image).is_file(),image
        checks.update(report_checked=True,report_bytes=(REPORT/'report.html').stat().st_size,actual_figure_download_anchors=3*len(records),manuscript_image_links='resolved',sources=len(artifact['manifest']['sources']),charts=len(artifact['manifest']['charts']),tables=len(artifact['manifest']['tables']),browser_interaction='not tested in this environment')
    if args.bundle:
        assert args.verify_report,'Bundle only after report checks.'
        path=REPORT/'report_bundle.zip'
        with ZipFile(path,'w',compression=ZIP_DEFLATED) as bundle:
            for name in ['report.html','artifact.json','README.md','build_receipt.json','build_notes.json','report_data.sqlite']:
                bundle.write(REPORT/name,name)
            for file in sorted((REPORT/'figures').glob('*')):
                if file.is_file():bundle.write(file,'figures/'+file.name)
            bundle.write(ROOT/'docs/icra2027/draft.md','docs/icra2027/draft.md')
            for row in records:
                file=FIG/f'{row["id"]}.svg';bundle.write(file,'docs/icra2027/figures/unified20/'+file.name)
        checks.update(bundle_bytes=path.stat().st_size,bundle='report_bundle.zip')
    write(OUT/'delivery_checks.json',checks);print(checks)


if __name__=='__main__':main()
