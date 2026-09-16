#!/usr/bin/env python3
"""Replace section 3.6 in bilingual v3 DOCX, preserving surrounding OOXML.

Native Word equations and tables are produced from the reviewed Chinese
section. Only 3.6, the moved reversal discussion in 3.4, and the later table
number change. Work products and original backup stay in the report folder.
Run this after the analytical figures and section36.md have been finalized.
"""
from pathlib import Path
import unicodedata, re, subprocess
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root=Path(__file__).resolve().parents[2]
target=root/'docs/icra2027/中文论文/2026_09_14-初稿中英对照-v3.docx'
report=root/'workspace/reports/section36_rewrite_20260915'
work=Path('/tmp/docx_section36_rewrite_20260915');work.mkdir(parents=True,exist_ok=True)
backup=report/'v3_before_section36_rewrite.docx'
if not backup.exists():backup.write_bytes(target.read_bytes())
source=(report/'section36.md').read_text()
assert '<!-- PATH_ANALYSIS -->' not in source
# Only presentation conversion: equivalent notation and line wrapping.
equation_numbers=[int(n) for n in re.findall(r'\\tag\{(\d+)\}',source)]
assert equation_numbers==list(range(11,11+len(equation_numbers)))
source=source.replace(r'\begin{aligned}',r'\begin{gathered}').replace(r'\end{aligned}',r'\end{gathered}')
source=source.replace('&=','=').replace(r'&\approx',r'\approx')
source=re.sub(r'\\tag\{\d+\}', '',source)
source=re.sub(r'\$\$(.*?)\$\$',lambda m:'$$\n'+m.group(1).strip()+'\n$$',source,flags=re.S)
source=source.replace(r'\{1,2\}',r'\left\{1,2\right\}')
# Transposed bracket vectors must be genuine math delimiters in Word/LibreOffice.
source=source.replace(r'[\cos\bar\theta_i,\sin\bar\theta_i]^\top',r'\left[\cos\bar\theta_i,\sin\bar\theta_i\right]^\top')
source=source.replace(r'[-\sin\bar\theta_i,\cos\bar\theta_i]^\top',r'\left[-\sin\bar\theta_i,\cos\bar\theta_i\right]^\top')
source=source.replace(r'[\alpha_1^{\ell+1},\ldots,\alpha_M^{\ell+1}]^\top',r'\left[\alpha_1^{\ell+1},\ldots,\alpha_M^{\ell+1}\right]^\top')
def image_link(m):
 p=(report/m.group(2)).resolve();assert p.is_file(),p
 return '![]('+str(p)+')'
source=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',image_link,source)
(work/'section.md').write_text(source)
subprocess.run(['pandoc',str(work/'section.md'),'--from=markdown+tex_math_dollars','--to=docx','--reference-doc='+str(backup),'-o',str(work/'section.docx')],check=True)
frag=Document(work/'section.docx')
num=11
for p in frag.paragraphs:
 heading=p.style.name.startswith('Heading')
 p.style=frag.styles['Heading 2' if heading else 'Normal']
 p.paragraph_format.space_after=Pt(6)
 p.paragraph_format.keep_together=True
 if heading:
  p.paragraph_format.keep_with_next=True
  p.paragraph_format.space_before=Pt(9)
 if p.text.rstrip().endswith(('得到','求解','有','写为','近似：','评价：')):
  p.paragraph_format.keep_with_next=True
 if p.text.startswith('【表'):
  p.paragraph_format.keep_with_next=True
 if p._p.xpath('.//w:drawing'):
  p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  p.paragraph_format.keep_with_next=True
 if p._p.xpath('.//m:oMathPara'):
  # Use native editable Office Math, matching the original document's inline
  # display layout, with the equation number in an ordinary text run.
  for ompara in p._p.xpath('./m:oMathPara'):
   idx=p._p.index(ompara)
   for math in list(ompara):
    if math.tag==qn('m:oMath'):
     p._p.insert(idx,math);idx+=1
   p._p.remove(ompara)
  p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  p.add_run(f'   ({num})')
  p.paragraph_format.space_before=Pt(5)
  p.paragraph_format.space_after=Pt(7)
  num+=1
 for r in p.runs:
  fonts=r._r.get_or_add_rPr().find(qn('w:rFonts'))
  if fonts is None:
   fonts=OxmlElement('w:rFonts');r._r.get_or_add_rPr().append(fonts)
  fonts.set(qn('w:eastAsia'),'Songti SC')
 for acc in p._p.xpath('.//m:acc'):
  chrnode=acc.find(qn('m:accPr'))
  chrnode=None if chrnode is None else chrnode.find(qn('m:chr'))
  if chrnode is not None and chrnode.get(qn('m:val'))in ['̅','‾','̄']:
   bar=OxmlElement('m:bar');pr=OxmlElement('m:barPr');pos=OxmlElement('m:pos');pos.set(qn('m:val'),'top');pr.append(pos);bar.append(pr)
   bar.append(deepcopy(acc.find(qn('m:e'))));acc.getparent().replace(acc,bar)
 for mr in p._p.xpath('.//m:r'):
  mathpr=mr.find(qn('m:rPr'))
  if mathpr is None:
   mathpr=OxmlElement('m:rPr');mr.insert(0,mathpr)
  sty=mathpr.find(qn('m:sty'))
  mode=sty.get(qn('m:val')) if sty is not None else 'i'
  if mode=='p' and mathpr.find(qn('m:nor')) is None: mathpr.append(OxmlElement('m:nor'))
  if 'b' in mode:
   for token in mr.findall(qn('m:t')):
    converted=''
    for ch in token.text or '':
     label=unicodedata.name(ch,'')
     try:
      if label.startswith('LATIN '): ch=unicodedata.lookup('MATHEMATICAL BOLD '+label.removeprefix('LATIN ').replace('LETTER ',''))
      elif label.startswith('GREEK '): ch=unicodedata.lookup('MATHEMATICAL BOLD ITALIC '+label.removeprefix('GREEK ').replace('LETTER ',''))
     except KeyError: pass
     converted+=ch
    token.text=converted
  rpr=mr.find(qn('w:rPr'))
  if rpr is None:
   rpr=OxmlElement('w:rPr');mr.insert(1 if mr.find(qn('m:rPr')) is not None else 0,rpr)
  fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Cambria Math');fonts.set(qn('w:hAnsi'),'Cambria Math');rpr.append(fonts)
  size=OxmlElement('w:sz');size.set(qn('w:val'),'21');rpr.append(size)
  for key,val in [('w:b','1' if 'b' in mode else '0'),('w:i','1' if 'i' in mode else '0')]:
   el=OxmlElement(key);el.set(qn('w:val'),val);rpr.append(el)
assert num==11+len(equation_numbers)
for image_i,shape in enumerate(frag.inline_shapes):
 ratio=shape.height/shape.width
 shape.width=Cm(16.5);shape.height=int(shape.width*ratio)
for table in frag.tables:
 table.autofit=False
 borders=OxmlElement('w:tblBorders')
 for edge in ['top','bottom','left','right','insideH','insideV']:
  border=OxmlElement('w:'+edge);border.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil');border.set(qn('w:sz'),'6');border.set(qn('w:color'),'666666');borders.append(border)
 table._tbl.tblPr.append(borders)
 for cell in table.rows[0].cells:
  borders=OxmlElement('w:tcBorders');bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'single');bottom.set(qn('w:sz'),'4');bottom.set(qn('w:color'),'888888');borders.append(bottom);cell._tc.get_or_add_tcPr().append(borders)
 for col,width in zip(table.columns,[6.8,5.1,5.1]):col.width=Cm(width)
 for row in table.rows:
  prop=row._tr.get_or_add_trPr();prop.append(OxmlElement('w:cantSplit'))
  for cell,width in zip(row.cells,[6.8,5.1,5.1]):
   cell.width=Cm(width)
   for p in cell.paragraphs:
    p.style=frag.styles['Normal']
    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
    for run in p.runs:run.font.size=Pt(10)
 table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
frag.save(work/'formatted.docx')

ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
rel_ns='http://schemas.openxmlformats.org/package/2006/relationships'
ct_ns='http://schemas.openxmlformats.org/package/2006/content-types'
with ZipFile(backup) as z: original={i.filename:z.read(i.filename) for i in z.infolist()}
with ZipFile(work/'formatted.docx') as z: extra={i.filename:z.read(i.filename) for i in z.infolist()}
doc=E.fromstring(original['word/document.xml']);body=doc.find('w:body',ns)
fragment=E.fromstring(extra['word/document.xml']).find('w:body',ns)
rels=E.fromstring(original['word/_rels/document.xml.rels'])
frels=E.fromstring(extra['word/_rels/document.xml.rels'])
def text(el):return ''.join(el.xpath('.//w:t/text()',namespaces=ns))
start=next(i for i,x in enumerate(body) if text(x).startswith('3.6') and '历史' in text(x))
end=next(i for i,x in enumerate(body) if text(x).startswith('3.7') and '实机' in text(x))
following=body[end]
oldsection=list(body)[start:end]
old_image_ids={v for child in oldsection for elem in child.iter() for attr,v in elem.attrib.items() if attr==qn('r:embed')}
removed_bookmarks={elem.get(qn('w:id')) for child in oldsection for elem in child.iter() if elem.tag in [qn('w:bookmarkStart'),qn('w:bookmarkEnd')]}
for child in oldsection:body.remove(child)
for elem in list(body):
 if elem.tag in [qn('w:bookmarkStart'),qn('w:bookmarkEnd')] and elem.get(qn('w:id')) in removed_bookmarks:body.remove(elem)
start=body.index(following)
# Remove the reversal-specific tail of the two bilingual paragraphs in 3.4;
# preserve their preceding spatial-region result and all its inline equations.
moved=[]
for child in list(body)[:start]:
 for marker in ['为进一步考察模型对加载路径变化的响应','To further examine responses to changes in loading path']:
  for node in child.xpath('.//w:t',namespaces=ns):
   value=node.text or ''
   if marker not in value:continue
   node.text=value.split(marker,1)[0].rstrip()
   rootchild=node
   while rootchild.getparent() is not child:rootchild=rootchild.getparent()
   trailing=list(child)[child.index(rootchild)+1:]
   for x in trailing:child.remove(x)
   moved.append(marker)
   break
assert len(moved)==2,moved
for child in list(body)[start:]:
 for node in child.xpath('.//w:t',namespaces=ns):
  node.text=(node.text or '').replace('表6','表5').replace('Table 6','Table 5')
# Old pictures belonged only to replaced section; remove their unused parts.
remove_entries=set()
for rel in list(rels):
 if rel.get('Id') in old_image_ids:
  rels.remove(rel);remove_entries.add('word/'+rel.get('Target'))
ids={x.get('Id') for x in rels};mapping={};media={}
for r in frels:
 if r.get('Type','').endswith('/image'):
  rid=1
  while f'rId{rid}' in ids:rid+=1
  newrid=f'rId{rid}';ids.add(newrid);mapping[r.get('Id')]=newrid
  fpath='word/'+r.get('Target');name='section36_v3_rewrite_'+Path(fpath).name
  newpath='word/media/'+name;assert newpath not in original
  media[newpath]=extra[fpath]
  rel=E.SubElement(rels,'{'+rel_ns+'}Relationship')
  rel.set('Id',newrid);rel.set('Type',r.get('Type'));rel.set('Target','media/'+name)
children=[]
docpr_max=max([int(x.get('id')) for x in doc.xpath('//wp:docPr',namespaces=ns)]+[0])
for child in fragment:
 if child.tag in [qn('w:sectPr'),qn('w:bookmarkStart'),qn('w:bookmarkEnd')]:continue
 child=deepcopy(child)
 for b in child.xpath('.//w:bookmarkStart | .//w:bookmarkEnd',namespaces=ns):b.getparent().remove(b)
 for elem in child.iter():
  for attr,value in list(elem.attrib.items()):
   if attr.startswith('{'+ns['r']+'}'):
    assert value in mapping,(attr,value)
    elem.set(attr,mapping[value])
 for elem in child.xpath('.//wp:docPr',namespaces=ns):
  docpr_max+=1;elem.set('id',str(docpr_max))
 children.append(child)
for offset,child in enumerate(children):body.insert(start+offset,child)
ct=E.fromstring(original['[Content_Types].xml'])
if not any(x.get('Extension')=='png' for x in ct):
 el=E.SubElement(ct,'{'+ct_ns+'}Default');el.set('Extension','png');el.set('ContentType','image/png')
def serial(x):return E.tostring(x,encoding='UTF-8',xml_declaration=True,standalone=True)
changes={'word/document.xml':serial(doc),'word/_rels/document.xml.rels':serial(rels),'[Content_Types].xml':serial(ct),**media}
output=report/'v3_section36_review.docx'
with ZipFile(output,'w',ZIP_DEFLATED) as z:
 for name,data in original.items():
  if name not in remove_entries:z.writestr(name,changes.pop(name,data))
 for name,data in changes.items():z.writestr(name,data)
check=Document(output)
assert len(check.inline_shapes)==2
assert len(check.tables)==9
assert len([p for p in check.paragraphs if p.text.strip() in [f'({n})' for n in equation_numbers]])==len(equation_numbers)
print(output)
print('Replaced 3.6; moved reversal discussion from 3.4; 2 figures, 1 new table,',len(equation_numbers),'editable numbered equations.')
