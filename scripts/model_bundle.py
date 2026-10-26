"""Pack, verify and install self-contained inference models."""
import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from storage import require_output, work_dir

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["O"] + [prefix + kind for kind in ["PERSON", "ADDRESS", "PHONE", "EMAIL", "ID", "DATE", "HOSPITAL"] for prefix in ["B-", "I-"]]
FILES = {"best.bpk", "burn-config.json", "tokenizer.json", "labels.json", "MODEL_CARD.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "licenses/dependency-index.json", "licenses/bert-burn-MIT.txt"}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('runtime_version') != '0.1.0' or manifest.get('format') != 'burn-bpk-rbt3-token-classifier':
        raise ValueError('unsupported model manifest')
    if set(manifest['files']) != FILES:
        raise ValueError('incomplete model bundle')
    for name, record in manifest['files'].items():
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'model bundle integrity check failed: {name}')
    if json.loads((directory / 'labels.json').read_text()) != LABELS:
        raise ValueError('model labels differ from this runtime')
    return manifest


def pack(checkpoint, model_dir, model_card, output, model_id):
    require_output(output)
    if output.exists():
        raise ValueError('model archive already exists; choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = work_dir() / 'tmp'
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as staging:
        directory = Path(staging)
        for source, name in [(checkpoint,'best.bpk'),(model_dir/'burn-config.json','burn-config.json'),
                             (model_dir/'tokenizer.json','tokenizer.json'),(model_card,'MODEL_CARD.md'),
                             (ROOT/'LICENSE','LICENSE'),(ROOT/'NOTICE','NOTICE'),(ROOT/'THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md'),
                             (ROOT/'licenses/dependency-index.json','licenses/dependency-index.json'),
                             (ROOT/'licenses/bert-burn-MIT.txt','licenses/bert-burn-MIT.txt')]:
            (directory/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source, directory/name)
        (directory/'labels.json').write_text(json.dumps(LABELS,indent=2)+'\n')
        manifest = {'schema_version':1,'model_id':model_id,'runtime_version':'0.1.0',
                    'format':'burn-bpk-rbt3-token-classifier','license':'Apache-2.0',
                    'base_model':'hfl/rbt3','base_revision':'0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c',
                    'offset_unit':'utf8_bytes_end_exclusive','files':{}}
        for name in sorted(FILES):
            path=directory/name
            manifest['files'][name]={'bytes':path.stat().st_size,'sha256':digest(path)}
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        verify(directory)
        partial=output.with_suffix(output.suffix+'.part')
        try:
            with partial.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped, tarfile.open(fileobj=zipped,mode='w') as archive:
                for name in sorted(FILES | {'manifest.json'}):
                    content=(directory/name).read_bytes()
                    info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644
                    archive.addfile(info,io.BytesIO(content))
            partial.rename(output)
        finally:
            partial.unlink(missing_ok=True)
    checksum=digest(output)
    output.with_name(output.name+'.sha256').write_text(f'{checksum}  {output.name}\n')
    return {'archive':str(output),'sha256':checksum,'model_id':model_id}


def install(archive_path, destination, expected_hash):
    require_output(destination)
    if destination.exists():
        raise ValueError('model destination already exists')
    if digest(archive_path) != expected_hash.lower():
        raise ValueError('archive checksum mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        directory=Path(staging)
        with tarfile.open(archive_path,'r:gz') as archive:
            seen=set()
            for member in archive:
                limit = 512 * 1024**2 if member.name == 'best.bpk' else 32 * 1024**2 if member.name == 'tokenizer.json' else 1024**2
                if member.name not in FILES | {'manifest.json'} or member.name in seen or not member.isfile() or member.size < 0 or member.size > limit:
                    raise ValueError('unsafe or unsupported model archive member')
                seen.add(member.name)
                stream=archive.extractfile(member)
                if stream is None: raise ValueError('unreadable archive member')
                (directory/member.name).parent.mkdir(parents=True,exist_ok=True)
                with stream, (directory/member.name).open('wb') as output:
                    shutil.copyfileobj(stream,output)
        if seen != FILES | {'manifest.json'}:
            raise ValueError('incomplete archive')
        manifest=verify(directory)
        directory.rename(destination)
    return {'model_id':manifest['model_id'],'directory':str(destination)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('pack')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--model-card',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-id',required=True)
    p=commands.add_parser('install')
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sha256',required=True)
    p=commands.add_parser('verify');p.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='pack': result=pack(args.checkpoint,args.model_dir,args.model_card,args.output,args.model_id)
    elif args.command=='install': result=install(args.archive,args.output,args.sha256)
    else: result=verify(args.directory)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
