"""Build the portable validation App from runtime files and explicit models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from ..runtime.hereditary_deployment import load_bundle


def build_package(bundle, out, candidates_dir=None, sam2_source=None,
                  sam2_checkpoint=None, perception_dir=None, yolo_license=None):
    root = Path(__file__).resolve().parents[1]
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    entries, models, perception, third_party = {}, [], [], []

    def add_bytes(data, name):
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name:
            raise ValueError('Archive path must be a normalized relative path')
        if name in entries:
            raise ValueError('Duplicate archive path: '+name)
        entries[name] = data

    def add(source, name):
        add_bytes(Path(source).read_bytes(), name)

    def add_json(value, name):
        add_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode(), name)

    def model(source, name, role):
        source = Path(source)
        _, meta = load_bundle(source)
        if any(m['weights_sha256'] == meta['weights_sha256'] for m in models):
            return
        add(source, name+'.npz')
        # Source IDs/hashes identify the model; a server absolute path is not a dependency.
        meta.pop('source_checkpoint', None)
        add_json(meta, name+'.json')
        models.append(dict(path=name+'.npz', role=role, dt=meta['dt'],
                           weights_sha256=meta['weights_sha256']))

    for path in sorted(root.rglob('*.py')):
        rel = path.relative_to(root)
        if any(x in rel.parts for x in ('__pycache__', 'checkpoints', 'runs', 'data', 'vendor', 'tools', 'portable')):
            continue
        add(path, 'real_validation/'+rel.as_posix())
    add(root/'PORTABLE_README.md', 'README.md')
    for name in ('install.bat', 'run.bat'):
        add(root/'portable'/name, name)
    add(root/'config/hardware.example.json', 'real_validation/config/hardware.example.json')
    model(bundle, 'real_validation/checkpoints/hereditary_current/hereditary', 'default')
    if candidates_dir:
        for candidate in sorted(Path(candidates_dir).glob('*.npz')):
            model(candidate, 'real_validation/checkpoints/candidates/'+candidate.stem, 'candidate')

    requirements = ['requirements.txt', 'requirements-hardware.txt']
    if bool(sam2_source) != bool(sam2_checkpoint):
        raise ValueError('SAM2 source and checkpoint must be provided together')
    if sam2_source:
        sam2_source = Path(sam2_source)
        if not (sam2_source/'sam2/build_sam.py').is_file():
            raise ValueError('Invalid upstream SAM2 source')
        for path in sorted((sam2_source/'sam2').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.yaml', '.yml'):
                add(path, 'real_validation/vendor/sam2/'+path.relative_to(sam2_source).as_posix())
        for name in ('LICENSE', 'LICENSE_cctorch'):
            add(sam2_source/name, 'real_validation/vendor/sam2/'+name)
        add(sam2_checkpoint, 'real_validation/checkpoints/sam2/sam2.1_hiera_tiny.pt')
        third_party.append(dict(name='SAM2', source='https://github.com/facebookresearch/sam2', license='Apache-2.0',
                                checkpoint_sha256=hashlib.sha256(Path(sam2_checkpoint).read_bytes()).hexdigest()))
        requirements.append('requirements-sam2.txt')
    if perception_dir:
        from ..perception.yolo_initial import load_contract
        contracts = sorted(Path(perception_dir).glob('*/inference.json'))
        if not contracts:
            raise ValueError('No YOLO candidate contracts found')
        if not yolo_license:
            raise ValueError('YOLO distribution license required')
        for contract in contracts:
            config = load_contract(contract.parent)
            prefix = 'real_validation/checkpoints/perception/'+contract.parent.name+'/'
            config['files'] = {'best.pt':config['files']['best.pt']}
            config.pop('onnx', None)
            add(contract.parent/'best.pt', prefix+'best.pt')
            add_json(config, prefix+'inference.json')
            provenance = json.loads((contract.parent/'provenance.json').read_text())
            perception.append(dict(path=prefix+'inference.json', method='yolo26_seg',
                                   weights_sha256=config['files']['best.pt'],
                                   study_id=provenance['study_id'], selection=provenance['selection']))
        add(yolo_license, 'real_validation/licenses/Ultralytics-LICENSE.txt')
        third_party.append(dict(name='Ultralytics YOLO', version='8.4.146', license='AGPL-3.0',
                                source='https://github.com/ultralytics/ultralytics'))
        requirements.append('requirements-yolo.txt')
    # One install list, assembled from the same dependencies used in the source App.
    lines = []
    for name in requirements:
        for line in (root/name).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(('#', '-r ')) and line not in lines:
                lines.append(line)
    add_bytes(('\n'.join(lines)+'\n').encode(), 'real_validation/requirements.txt')
    manifest = dict(schema='hereditary_portable_package_v3', models=models,
                    perception_models=perception, third_party=third_party,
                    files={name:hashlib.sha256(data).hexdigest() for name, data in entries.items()})
    add_json(manifest, 'real_validation/PACKAGE_MANIFEST.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--candidates-dir', type=Path)
    p.add_argument('--sam2-source', type=Path)
    p.add_argument('--sam2-checkpoint', type=Path)
    p.add_argument('--perception-dir', type=Path)
    p.add_argument('--yolo-license', type=Path)
    a = p.parse_args()
    print(build_package(**vars(a)))


if __name__ == '__main__':
    main()
