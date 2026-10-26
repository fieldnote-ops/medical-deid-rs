#!/usr/bin/env python3
"""Download fixed public weights; convert locally and emit synthetic reference outputs."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import argparse
from storage import require_output

REVISION = "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    require_output(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    endpoints=list(dict.fromkeys([os.environ.get("HF_ENDPOINT","https://hf-mirror.com").rstrip("/"),"https://huggingface.co"]))
    def fetch(relative,target):
        for endpoint in endpoints:
            try:
                subprocess.run(["curl","--fail","--location","--silent","--show-error","--retry","2","--connect-timeout","15","--max-time","240",f"{endpoint}/{relative}","--output",str(target)],check=True)
                return endpoint
            except subprocess.CalledProcessError:
                print(f"Download failed at {endpoint}; trying next configured source",flush=True)
        raise RuntimeError("all configured public model sources failed")
    # Resolve and record immutable revision, rather than trusting an unverified hard-coded hash.
    provenance_path=args.output/"provenance.json"
    previous={}
    if provenance_path.exists():
        previous=json.loads(provenance_path.read_text())
        revision=previous["revision"]
    else:
        source_path=args.output/"source-metadata.json"
        fetch(f"api/models/hfl/rbt3/revision/{REVISION}?blobs=true",source_path)
        metadata=json.loads(source_path.read_text())
        revision=REVISION
    if revision != REVISION:
        raise RuntimeError("local pretrained model revision differs from this release")
    files=["config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt","pytorch_model.bin"]
    hashes={}
    sources=previous.get("download_sources",{})
    for filename in files:
        target=args.output/filename
        if not target.exists():
            print(f"Downloading public model artifact: {filename}",flush=True)
            sources[filename]=fetch(f"hfl/rbt3/resolve/{revision}/{filename}",target.with_suffix(target.suffix+".part"))
            target.with_suffix(target.suffix+".part").rename(target)
        hashes[filename]=hashlib.sha256(target.read_bytes()).hexdigest()
        if filename in previous.get("sha256",{}) and hashes[filename]!=previous["sha256"][filename]:
            raise RuntimeError(f"cached artifact changed: {filename}")
        sources.setdefault(filename,"existing_local_cache")
    source_path=args.output/"source-metadata.json"
    if source_path.exists():
        for item in json.loads(source_path.read_text()).get("siblings",[]):
            expected=item.get("lfs",{}).get("sha256")
            if expected and item["rfilename"] in hashes:
                if hashes[item["rfilename"]]!=expected:raise RuntimeError("public weight hash mismatch")
    import torch
    if hashes["pytorch_model.bin"] != "3e04f7477f55dffce2a2fbc4d0ba35068415162a9e92e3d5cc74a49781ba4eb0":
        raise RuntimeError("pretrained checkpoint checksum mismatch")
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    state=torch.load(args.output/"pytorch_model.bin",map_location="cpu",weights_only=True)
    save_file({k:v.contiguous().clone() for k,v in state.items()},str(args.output/"model.safetensors"))
    config=json.loads((args.output/"config.json").read_text())
    config.update(pad_token_id=0,model_type="bert",max_seq_len=128,with_pooling_layer=False)
    (args.output/"burn-config.json").write_text(json.dumps(config,indent=2)+"\n")
    tokenizer=AutoTokenizer.from_pretrained(args.output,local_files_only=True)
    model=AutoModel.from_pretrained(args.output,local_files_only=True,attn_implementation="eager").eval()
    texts=["合成示例：患者林清禾，肌酐88 μmol/L。","🙂联系人周明远，电话13800000000。"]
    encoded=tokenizer(texts,padding=True,return_tensors="pt")
    with torch.no_grad(): out=model(**encoded).last_hidden_state
    reference=dict(input_ids=encoded["input_ids"].tolist(),attention_mask=encoded["attention_mask"].tolist(),hidden_states=out.tolist())
    (args.output/"reference.json").write_text(json.dumps(reference)+"\n")
    hashes["model.safetensors"]=hashlib.sha256((args.output/"model.safetensors").read_bytes()).hexdigest()
    (args.output/"provenance.json").write_text(json.dumps(dict(repository="hfl/rbt3",revision=revision,sha256=hashes,download_sources=sources,preferred_endpoint=endpoints[0],scope="public_pretrained_model_and_synthetic_reference"),indent=2)+"\n")
    print(json.dumps(dict(revision=revision,reference_shape=list(out.shape),parameters=sum(x.numel() for x in model.parameters())),indent=2))

if __name__=="__main__": main()
