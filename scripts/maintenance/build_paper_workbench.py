#!/usr/bin/env python3
"""Build an offline paper-preparation reader from governed Markdown and catalog.

Usage: python scripts/maintenance/build_paper_workbench.py --out workspace/reports/<unique>/index.html
Requires Markdown and beautifulsoup4. Refuses to overwrite an existing output.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'docs/paper/occlusion_control'
REVIEW = ROOT / 'docs/papers/review_partial_observation_20260910'


def build(output: Path) -> None:
    import markdown
    from bs4 import BeautifulSoup

    catalog = json.loads((REVIEW / 'catalog.json').read_text(encoding='utf-8'))
    paths = [PAPER / 'README.md', *sorted(PAPER.glob('0[1-5]_*.md')),
             REVIEW / 'README.md', REVIEW / 'core_papers.md',
             REVIEW / 'introduction_atlas.md', REVIEW / 'evidence_corrections.md']
    labels = ['阅读路线', '关键点与证据', '引言与故事', '方法与公式', '实验与对照',
              '写作与展望', '文献分类说明', '核心论文卡片', '引言逐段图谱', '文献纠错']
    ids = {p.resolve(): f'doc{i}' for i, p in enumerate(paths)}
    docs = []
    for path, label in zip(paths, labels):
        raw = path.read_text(encoding='utf-8')
        raw = re.sub(r'\A---\n.*?\n---\n', '', raw, flags=re.S)
        formulas: list[str] = []

        def formula(match: re.Match) -> str:
            idx = len(formulas)
            block = match.group(0).startswith(r'\[')
            tag = 'pre' if block else 'code'
            formulas.append(f'<{tag} class="math">{html.escape(match.group(0))}</{tag}>')
            return f'MATHPLACEHOLDER{idx}END'

        raw = re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)', formula, raw, flags=re.S)
        rendered = markdown.markdown(raw, extensions=['tables', 'fenced_code', 'toc'])
        for i, value in enumerate(formulas):
            rendered = rendered.replace(f'<p>MATHPLACEHOLDER{i}END</p>', value)
            rendered = rendered.replace(f'MATHPLACEHOLDER{i}END', value)
        soup = BeautifulSoup(rendered, 'html.parser')
        for link in soup.find_all('a', href=True):
            target = link['href']
            parsed = urlsplit(target)
            if parsed.scheme:
                link['target'] = '_blank'
                link['rel'] = 'noopener noreferrer'
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path.resolve()
            if resolved in ids:
                link['href'] = '#' + ids[resolved] + ('/' + quote(parsed.fragment) if parsed.fragment else '')
            elif resolved == REVIEW / 'catalog.md':
                link['href'] = '#catalog'
            else:
                link['href'] = quote(os.path.relpath(resolved, output.parent)) + ('#' + parsed.fragment if parsed.fragment else '')
                link['target'] = '_blank'
        for table in soup.find_all('table'):
            wrap = soup.new_tag('div', attrs={'class': 'table-scroll', 'tabindex': '0'})
            table.wrap(wrap)
        docs.append({'id': ids[path.resolve()], 'label': label, 'body': str(soup)})
    payload = json.dumps({'catalog': catalog, 'docs': docs,
                          'csv': (REVIEW / 'catalog.csv').read_text(encoding='utf-8-sig')}, ensure_ascii=False).replace('<', '\\u003c')
    page = TEMPLATE.replace('__PAYLOAD__', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(page)
    print(f'{output}: {len(catalog["records"])} records, {len(docs)} documents, {len(page.encode())} bytes')


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>部分观测闭环控制 · 论文准备工作台</title>
<style>
:root{--ink:#182b33;--muted:#526670;--line:#d6e0e1;--accent:#006966;--paper:#fff;--bg:#f3f6f5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.75 system-ui,-apple-system,"Microsoft YaHei",sans-serif}a{color:var(--accent);text-underline-offset:3px}button,input,select{font:inherit}button,select,input{border:1px solid #879e9f;border-radius:5px;padding:8px 12px;background:white;color:var(--ink)}button{cursor:pointer}button:hover{background:#e4f1ed}:focus-visible{outline:3px solid #cb7c23;outline-offset:3px}.skip{position:absolute;left:-999px}.skip:focus{left:10px;top:10px;background:white;padding:10px;z-index:5}header{padding:32px max(24px,calc((100vw - 1440px)/2));background:#153c40;color:white}header small{color:#c6ddda;letter-spacing:.08em}header h1{font-size:30px;line-height:1.35;margin:8px 0 16px}header p{max-width:1050px;margin:8px 0;color:#e1edeb}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:20px}.flow span{background:#285154;border:1px solid #719795;padding:9px 15px;border-radius:5px}.flow b{color:#a8dbcb}.layout{max-width:1490px;margin:auto;display:grid;grid-template-columns:220px minmax(0,1fr);gap:24px;padding:24px}nav{align-self:start;position:sticky;top:18px}nav a{display:block;padding:7px 12px;border-radius:4px;margin:2px 0;text-decoration:none;color:var(--ink)}nav a.active{background:#dbeee7;font-weight:700;color:#005f51}nav small{display:block;color:var(--muted);padding:6px 12px}.content{background:white;border:1px solid var(--line);border-radius:8px;padding:28px;min-width:0}.badge{display:inline-block;border-radius:4px;padding:1px 7px;font-size:12px;background:#e3f0e8;color:#254f3e;white-space:nowrap}.badge.pending{background:#fff0d6;color:#6b481b}.badge.note{background:#edf0f2;color:#46575e}.filters{display:grid;grid-template-columns:2fr 1.2fr 1fr;gap:12px;margin:18px 0}.filters label{display:flex;flex-direction:column;gap:5px;font-size:14px}.filters input,.filters select{width:100%;min-width:0}.tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}.count{font-variant-numeric:tabular-nums;font-weight:700;margin-right:auto}.table-scroll{overflow-x:auto;margin:16px 0;max-width:100%}table{border-collapse:collapse;font-size:14px;width:100%}th{text-align:left;background:#edf4f0;color:#31584b}th,td{border-bottom:1px solid var(--line);padding:12px;vertical-align:top}td:first-child{min-width:130px}td.title{min-width:260px}td p{margin:4px 0}td small{color:var(--muted)}details summary{cursor:pointer;color:#4c6368}details p{max-width:560px}.tags{font-size:12px;color:#386451}.notice{padding:12px 16px;background:#f3f6f3;border-left:3px solid #769c81;color:#435a52}.empty{text-align:center;padding:40px;color:var(--muted)}h1{font-size:27px;line-height:1.4}h2{font-size:22px;margin-top:32px}h3{font-size:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px;font:14px/1.75 ui-monospace,monospace}code{font:14px/1.6 ui-monospace,monospace;background:#f2f5f7;padding:2px 4px;overflow-wrap:anywhere}blockquote{margin-left:0;border-left:3px solid #91b3aa;padding-left:20px;color:#3f5e55}.doc-top{display:flex;gap:10px;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:12px;color:var(--muted);font-size:13px}li{margin:7px 0}footer{max-width:1200px;margin:0 auto;padding:0 24px 28px;color:var(--muted);font-size:13px}[hidden]{display:none!important}@media(max-width:900px){.layout{grid-template-columns:1fr;padding:12px;gap:12px}nav{position:static;display:flex;flex-wrap:wrap;gap:2px}nav small{width:100%}nav a{font-size:14px}header{padding:22px}header h1{font-size:25px}.content{padding:18px}.filters{grid-template-columns:1fr}.flow{font-size:13px;gap:6px}.flow span{padding:6px 8px}}@media print{header,nav,.filters,.tools,.doc-top,footer{display:none}.layout{display:block;padding:0}.content{border:0;padding:0}.table-scroll{overflow:visible}body{background:white;font-size:12px}h2{break-after:avoid}tr{break-inside:avoid}}
</style></head><body>
<a class="skip" href="#main">跳到内容</a>
<header><small>SELF SOFT ROBOT / RESEARCH NOTES / 2026-09-10</small><h1>部分观测闭环控制 · 论文准备工作台</h1>
<p>从可见证据修正历史状态，再检验修正能否改善下一段真实动作。当前范围：平面全身形状、单 RGB、固定模型参数、有期限的反馈。</p>
<div class="flow" role="img" aria-label="闭环机制：动作历史提供先验，局部图像校正状态，修订未执行动作，真实响应产生下一帧反馈"><span>动作历史 → 状态先验</span><b>＋</b><span>局部图像 → 状态校正</span><b>→</b><span>动作修订 → 真实响应 → 下一帧</span></div></header>
<div class="layout"><nav aria-label="工作台导航" id="nav"><small>论文准备 / 无需网络</small><a href="#catalog">文献搜索与筛选</a></nav>
<main id="main" class="content" tabindex="-1"><section id="catalog-view"><h1>文献索引</h1><p class="notice">71 条索引含 1 条重复别名；21 篇核对指定原文章节，8 篇逐段分析引言。其余为待核线索。不同平台的精度与频率不能直接排名。</p>
<div class="filters"><label>关键词（标题、标签、写作位置、边界）<input id="query" type="search" placeholder="例如：遮挡 / 历史 / AFT / 实时"></label><label>主分类<select id="cluster"><option value="">全部分类</option></select></label><label>核对等级<select id="verification"><option value="">全部等级</option></select></label></div>
<div class="tools"><span id="count" class="count" role="status" aria-live="polite"></span><button id="reset">清空筛选</button><button id="export-csv">导出全部 CSV</button><button id="export-json">导出全部 JSON</button></div>
<div class="table-scroll" tabindex="0"><table><thead><tr><th scope="col">记录 / 核对</th><th scope="col">文献 / 入口</th><th scope="col">主题与写作用途</th><th scope="col">引用边界</th></tr></thead><tbody id="rows"></tbody></table></div><p id="empty" class="empty" hidden>没有匹配记录。可缩短关键词或清空筛选。</p></section>
<section id="doc-view" hidden><div class="doc-top"><span>内嵌阅读版 · 公式保留可复制 LaTeX</span><button onclick="window.print()">打印 / 保存 PDF</button></div><article id="article"></article></section></main></div>
<footer>代码核对基准 96e684f。回放与虚拟设备是开发证据，真实遮挡闭环结果仍待验证。本文档与目录已内嵌；原论文链接需联网，代码/原始运行链接需保留仓库。原文核对范围与快照哈希见来源清单。</footer>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('payload').textContent), $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels=data.catalog.verification_labels;
for(const [id,label] of Object.entries(data.catalog.clusters)) $('cluster').add(new Option(id+' '+label,id));
for(const [id,label] of Object.entries(labels)) $('verification').add(new Option(label,id));
for(const doc of data.docs){const link=document.createElement('a');link.href='#'+doc.id;link.textContent=doc.label;$('nav').appendChild(link);}
const urlLink=(url,label)=>/^https?:\/\//.test(url)?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`:'';
function render(){const q=$('query').value.trim().toLowerCase().split(/\s+/).filter(Boolean),g=$('cluster').value,v=$('verification').value;
 const rows=data.catalog.records.filter(r=>(!g||r.cluster===g)&&(!v||r.verification===v)&&q.every(t=>[r.title,r.id,...r.tags,r.where_used,r.caveat,r.primary_url,r.cluster_name].join(' ').toLowerCase().includes(t)));
 $('count').textContent=`显示 ${rows.length} / ${data.catalog.records.length} 条`;$('empty').hidden=rows.length>0;
 $('rows').innerHTML=rows.map(r=>`<tr><td><strong>${esc(r.id)}</strong><br>${esc(r.year)} · ${esc(r.priority)}<br><span class="badge ${r.verification==='primary_checked'?'':r.verification==='note_only'?'note':'pending'}">${esc(labels[r.verification])}</span>${r.alias_of?'<br>主记录：'+esc(r.alias_of):''}</td><td class="title"><strong>${esc(r.title)}</strong><p>${urlLink(r.primary_url,'原文')}${r.core_card?' · <a href="#doc7/'+r.id+'">核心卡片</a>':''}${r.intro_atlas?' · <a href="#doc8/'+r.id+'">逐段引言</a>':''}</p></td><td><span class="tags">${esc(r.cluster+' · '+r.tags.join(' / '))}</span><p>${esc(r.where_used)}</p></td><td><details><summary>查看限制与核对范围</summary><p>${esc(r.caveat)}</p><small>${esc(r.metadata_basis)}</small></details></td></tr>`).join('');
}
for(const id of ['query','cluster','verification']) $(id).addEventListener(id==='query'?'input':'change',render);
$('reset').onclick=()=>{for(const id of ['query','cluster','verification']) $(id).value='';render();$('query').focus();};
function save(name,type,body){const u=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
$('export-csv').onclick=()=>save('literature_catalog_20260910.csv','text/csv;charset=utf-8','\ufeff'+data.csv);
$('export-json').onclick=()=>save('literature_catalog_20260910.json','application/json',JSON.stringify(data.catalog,null,2));
function route(){let [id,anchor]=(location.hash.slice(1)||'catalog').split('/');if(id==='main'){$('main').focus();return;}const doc=data.docs.find(d=>d.id===id);if(!doc)id='catalog';
 $('catalog-view').hidden=!!doc;$('doc-view').hidden=!doc;
 for(const a of $('nav').querySelectorAll('a')){const active=a.hash==='#'+id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');}
 if(doc){$('article').innerHTML=doc.body;document.title=doc.label+' · 论文准备工作台';if(anchor){const el=document.getElementById(decodeURIComponent(anchor));if(el)requestAnimationFrame(()=>el.scrollIntoView());}else window.scrollTo(0,0);}
 else {document.title='文献索引 · 论文准备工作台';render();}
}
window.addEventListener('hashchange',route);route();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.out.resolve())


if __name__ == '__main__':
    main()
