"""Create a clean portable workbench archive with one frozen model bundle."""
import argparse
from pathlib import Path
import zipfile
import re
from ..runtime.hereditary_deployment import load_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--guide',help='self-contained HTML operating guide to include')
    a=p.parse_args();bundle=Path(a.bundle);load_bundle(bundle)
    root=Path(__file__).resolve().parents[1];out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    guide=Path(a.guide) if a.guide else None
    if guide is not None and not guide.is_file():p.error('guide HTML not found')
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            rel=path.relative_to(root)
            if not path.is_file() or any(x in rel.parts for x in ('__pycache__','checkpoints','runs','data')):continue
            if path.suffix not in ('.py','.md','.txt','.bat','.sh') and rel.as_posix()!='config/hardware.example.json':continue
            if guide is not None and rel.as_posix()=='HEREDITARY_GUIDE.md':
                text=re.sub(r'\(\.\./workspace/[^)]+/guide.html\)', '(HEREDITARY_GUIDE.html)', path.read_text())
                archive.writestr('real_validation/'+rel.as_posix(),text)
            else:
                archive.write(path,'real_validation/'+rel.as_posix())
        if guide is not None:archive.write(guide,'real_validation/HEREDITARY_GUIDE.html')
        archive.write(bundle,'real_validation/checkpoints/hereditary_current/hereditary.npz')
        archive.write(bundle.with_suffix('.json'),'real_validation/checkpoints/hereditary_current/hereditary.json')
    print(out)
if __name__=='__main__':main()
