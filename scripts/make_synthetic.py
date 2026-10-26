#!/usr/bin/env python3
"""Deterministic fabricated records. Never reads clinical data."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from storage import require_output

def record(split, index, rng):
    first = {"train":"林周陈许宋沈", "dev":"赵吴郑王", "test":"冯蒋韩杨"}[split]
    given = "清禾明远安宁星舟语桐知夏若溪"
    name = lambda: rng.choice(first) + "".join(rng.choices(given, k=2))
    patient, relative = name(), name()
    address = f"虚构市示例区{rng.choice(['青禾','明月','清溪'])}路{rng.randrange(1,999)}号"
    hospital = f"示例{rng.choice(['青禾','明月','清溪'])}医院"
    phone = "138" + "".join(rng.choices("0123456789", k=8))
    email = f"fiction{split}{index}@example.invalid"
    date = f"2024-{rng.randrange(1,13):02d}-{rng.randrange(1,29):02d}"
    values = dict(person=patient,relative=relative,address=address,hospital=hospital,phone=phone,email=email,date=date,ident=f"SYN-{split.upper()}-{index:06d}")
    kinds = dict(person="PERSON",relative="PERSON",address="ADDRESS",hospital="HOSPITAL",phone="PHONE",email="EMAIL",date="DATE",ident="ID")
    templates = {
        "train":[
            ["姓名：", "{person}", "，家属", "{relative}", "。电话：", "{phone}", "。肌酐88 μmol/L，否认胸痛。"],
            ["患者", "{person}", "于", "{date}", "就诊，住址", "{address}", "。口服二甲双胍。"],
            ["联系人", "{relative}", "，邮箱", "{email}", "；病案号：", "{ident}", "。维生素B12正常。"],
            ["由", "{hospital}", "转入；患者", "{person}", "，居住于", "{address}", "。CT平扫未见异常。"],
            ["已向", "{relative}", "说明病情。手机号：", "{phone}", "，复查日期", "{date}", "。张力正常。"],
            ["签名：", "{person}", "；单位", "{hospital}", "；邮箱：", "{email}", "。血压120/80 mmHg。"],
        ],
        "dev":[
            ["随访记录：", "{person}", "，联络人", "{relative}", "；电话：", "{phone}", "。血糖5.6 mmol/L。"],
            ["来院日期", "{date}", "；居住地址：", "{address}", "，来自", "{hospital}", "。否认发热。"],
            ["病案号：", "{ident}", "；电子邮箱", "{email}", "；家属", "{relative}", "。药物剂量500 mg。"],
        ],
        "test":[
            ["本次访谈对象为", "{person}", "，其配偶", "{relative}", "陪同。联系方式：", "{phone}", "。否认高血压。"],
            ["转诊机构", "{hospital}", "；通信地址", "{address}", "；登记日期", "{date}", "。肌酐88 μmol/L。"],
            ["患者", "{person}", "留下邮箱", "{email}", "；住院号：", "{ident}", "。CT正常，体温36.5℃。"],
        ],
    }
    # Explicit PHI-free hard negatives are held apart by split and index.
    if index % 7 == 0:
        text = f"合成{split}病例{index}：张力正常，维生素B12、CA125及D-二聚体复查；血压120/80 mmHg。🙂"
        return dict(id=f"{split}-{index}",group=f"{split}-person-{index}",template=f"{split}-negative",text=text,spans=[])
    template_id = index % len(templates[split])
    text, spans = "", []
    for segment in templates[split][template_id]:
        if segment.startswith("{"):
            key = segment[1:-1]
            value = values[key]
            start = len(text.encode())
            text += value
            spans.append(dict(start=start,end=len(text.encode()),kind=kinds[key]))
        else:
            text += segment
    return dict(id=f"{split}-{index}",group=f"{split}-person-{index}",template=f"{split}-{template_id}",text=text,spans=spans)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train",type=int,default=384)
    parser.add_argument("--dev",type=int,default=64)
    parser.add_argument("--test",type=int,default=96)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    require_output(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(scope="synthetic_only",seed=args.seed,splits={})
    for split,count in [("train",args.train),("dev",args.dev),("test",args.test)]:
        rng=random.Random(f"{args.seed}-{split}")
        rows=[record(split,i,rng) for i in range(count)]
        content="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)
        (args.output/f"{split}.jsonl").write_text(content)
        manifest["splits"][split]=dict(documents=count,entities=sum(len(r["spans"]) for r in rows),sha256=hashlib.sha256(content.encode()).hexdigest())
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
