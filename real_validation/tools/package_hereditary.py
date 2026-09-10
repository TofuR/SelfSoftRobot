"""Create a portable workbench with verified model candidates and offline guides."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import zipfile
import re
from ..runtime.hereditary_deployment import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--guide',help='self-contained HTML operating guide to include')
    p.add_argument('--candidates-dir',type=Path,help='Explicit directory of named NPZ/JSON pairs and optional README.md')
    p.add_argument('--sam2-source',type=Path,help='Unmodified upstream SAM2 repository to bundle with its licenses')
    p.add_argument('--sam2-checkpoint',type=Path,help='Explicit SAM2.1 Tiny checkpoint for offline initialization')
    p.add_argument('--extra',nargs=2,action='append',default=[],metavar=('SOURCE','ARCHIVE_PATH'),help='Include an explicit supplementary document, image or evidence file')
    a=p.parse_args();bundle=Path(a.bundle);_,default_meta=load_bundle(bundle)
    root=Path(__file__).resolve().parents[1];out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    entries={};models=[]
    def add(source,name):
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name:
            raise ValueError('Archive path must be a normalized relative path')
        if name in entries:raise ValueError('Duplicate archive path: '+name)
        entries[name]=Path(source).read_bytes()
    def model(source,name,role):
        _,meta=load_bundle(source)
        add(source,name+'.npz');add(Path(source).with_suffix('.json'),name+'.json')
        models.append(dict(path=name+'.npz',role=role,dt=meta['dt'],weights_sha256=meta['weights_sha256']))
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if not path.is_file() or any(x in rel.parts for x in ('__pycache__','checkpoints','runs','data')):continue
        if path.suffix not in ('.py','.md','.txt','.bat','.sh') and rel.as_posix()!='config/hardware.example.json':continue
        add(path,'real_validation/'+rel.as_posix())
    model(bundle,'real_validation/checkpoints/hereditary_current/hereditary','default')
    if a.candidates_dir:
        candidates=sorted(a.candidates_dir.glob('*.npz'))
        if not candidates:raise ValueError('No model candidates found')
        for candidate in candidates:model(candidate,'real_validation/checkpoints/candidates/'+candidate.stem,'candidate')
        readme=a.candidates_dir/'README.md'
        if readme.is_file():add(readme,'real_validation/checkpoints/candidates/README.md')
    if bool(a.sam2_source)!=bool(a.sam2_checkpoint):raise ValueError('SAM2 source and checkpoint must be provided together')
    third_party=[]
    if a.sam2_source:
        if not (a.sam2_source/'sam2/build_sam.py').is_file():raise ValueError('Invalid upstream SAM2 source')
        for path in sorted((a.sam2_source/'sam2').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.yaml','.yml'):
                add(path,'real_validation/vendor/sam2/'+path.relative_to(a.sam2_source).as_posix())
        for name in ('LICENSE','LICENSE_cctorch','README.md'):
            add(a.sam2_source/name,'real_validation/vendor/sam2/'+name)
        add(a.sam2_checkpoint,'real_validation/checkpoints/sam2/sam2.1_hiera_tiny.pt')
        third_party.append(dict(name='SAM2',source='https://github.com/facebookresearch/sam2',license='Apache-2.0',
                                checkpoint_sha256=hashlib.sha256(a.sam2_checkpoint.read_bytes()).hexdigest()))
    if a.guide:
        add(a.guide,'real_validation/HEREDITARY_GUIDE.html')
        key='real_validation/HEREDITARY_GUIDE.md'
        text=entries[key].decode('utf-8')
        text=re.sub(r'\(\.\./workspace/[^)]+/guide.html\)', '(HEREDITARY_GUIDE.html)', text)
        entries[key]=text.encode('utf-8')
    for source,name in a.extra:add(source,name)
    # Standalone docs must not leave clickable links to absent server artifacts.
    # Keep their original location as text for provenance; bundled links stay live.
    for name,data in list(entries.items()):
        if not name.endswith('.md') or name.startswith('real_validation/vendor/'):continue
        def link(match):
            label,target=match.groups()
            if target.startswith(('#','https://','http://','mailto:')):return match.group(0)
            dest=posixpath.normpath(posixpath.join(posixpath.dirname(name),target.split('#')[0]))
            if dest in entries or any(key.startswith(dest.rstrip('/')+'/') for key in entries):return match.group(0)
            return label+'（源码仓库路径：`'+target+'`）'
        entries[name]=re.sub(r'(?<!!)\[([^]\n]+)\]\(([^)\n]+)\)',link,data.decode('utf-8')).encode('utf-8')
    manifest=dict(schema='hereditary_portable_package_v2',models=models,third_party=third_party,
                  files={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()})
    if 'PACKAGE_MANIFEST.json' in entries:raise ValueError('Reserved manifest path')
    entries['PACKAGE_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in entries.items():archive.writestr(name,data)
    print(out)
if __name__=='__main__':main()
