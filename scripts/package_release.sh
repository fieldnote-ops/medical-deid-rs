#!/usr/bin/env bash
set -euo pipefail
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo 'This release package targets Apple Silicon macOS.' >&2
  exit 1
fi
DEID_ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$DEID_ROOT/scripts/env.sh"
cd "$DEID_ROOT"
bash scripts/cargo.sh build --release --locked
python3 - <<'PY'
import hashlib, os, pathlib, shutil, tarfile, tempfile
root=pathlib.Path.cwd();work=pathlib.Path(os.environ['DEID_WORK'])
destination=work/'dist/medical-deid-rs-0.1.0-macos-arm64.tar.gz'
destination.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(dir=work/'tmp') as temporary:
    stage=pathlib.Path(temporary)/'medical-deid-rs';stage.mkdir()
    for name in ['README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','MODEL_CARD.md','CONTRIBUTING.md','SECURITY.md','CHANGELOG.md','.deid.env.example','docs','examples','licenses','model-manifests']:
        source=root/name
        if source.is_dir():shutil.copytree(source,stage/name)
        else:shutil.copy2(source,stage/name)
    (stage/'bin').mkdir();(stage/'scripts').mkdir()
    shutil.copy2(pathlib.Path(os.environ['CARGO_TARGET_DIR'])/'release/medical-deid',stage/'bin/medical-deid')
    for name in ['env.sh','deid.sh','storage.py','model.sh','model_bundle.py']:
        shutil.copy2(root/'scripts'/name,stage/'scripts'/name)
    with tarfile.open(destination,'w:gz') as archive:archive.add(stage,arcname=stage.name)
digest=hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_name(destination.name+'.sha256').write_text(f'{digest}  {destination.name}\n')
print(destination)
PY
