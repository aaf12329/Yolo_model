# -*- coding: utf-8 -*-
"""数据集一键下载器 / One-shot dataset downloader.

特性 / features:
- 断点续传（服务器不支持 Range 时自动整段重下）/ resumable (falls back to full download)
- 下载前后按制度检查 C 盘剩余容量 / C: free-space check per log.md policy
- 完成后按登记体积校验 / size verification against registered content-length
- --extract 解压（跳过 macOS 垃圾文件）/ optional extraction (skips macOS junk)

用法 / usage:
    python data/download_datasets.py                                  # 全部 / all
    python data/download_datasets.py --dataset mrl
    python data/download_datasets.py --dataset columbia --extract

被谁调用 / Called by: 手动 / manual
内部调用 / Calls: urllib, zipfile, shutil（仅标准库 / stdlib only）
数据来源与许可详见 / sources & licenses: data/README.md
"""
import argparse
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "datasets"

DATASETS = {
    "mrl": {
        "url": "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
        "zip": OUT_ROOT / "mrlEyes_2018_01.zip",
        "size": 341866898,
        "extract_to": OUT_ROOT,           # 解压出 mrlEyes_2018_01/
    },
    "columbia": {
        "url": "https://cave.cs.columbia.edu/old/databases/columbia_gaze/"
               "columbia_gaze_data_set.zip",
        "zip": OUT_ROOT / "columbia_gaze" / "columbia_gaze_data_set.zip",
        "size": 2378561401,
        "extract_to": OUT_ROOT / "columbia_gaze",  # 解压出 Columbia Gaze Data Set/
    },
}

JUNK = ("__MACOSX", ".DS_Store")


def check_c_free(tag: str):
    """制度要求：大动作前后查 C 盘剩余 / policy: check C: free space."""
    free = shutil.disk_usage("C:")[2] / 2**30
    print(f"[C盘检查 / disk check] {tag}: C 盘剩余 {free:.1f}G")
    return free


def download(name: str, cfg: dict):
    url, zip_path, total = cfg["url"], cfg["zip"], cfg["size"]
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    free = check_c_free("下载前 / before")
    if free * 2**30 < total * 1.6:
        raise SystemExit(f"C 盘空间不足（需约 {total*1.3/2**30:.1f}G），"
                         f"或按 log.md 策略改用 D:\\yolo_datasets\\")

    done = zip_path.stat().st_size if zip_path.exists() else 0
    if done >= total:
        print(f"[{name}] 已存在且大小相符，跳过下载 / already complete")
        return

    # Range 续传探测 / resume probe
    headers = {"Range": f"bytes={done}-"} if done else {}
    req = urllib.request.Request(url, headers=headers)
    print(f"[{name}] 从 {done/2**20:.0f}MB 续传 / resuming ..." if done
          else f"[{name}] 开始下载 / starting ...")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, \
         open(zip_path, "ab" if done and resp.status == 206 else "wb") as f:
        if done and resp.status != 206:
            print(f"[{name}] 服务器不支持断点续传（HTTP {resp.status}），整段重下 / full re-download")
            done = 0
            f.seek(0)
            f.truncate()
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
            if f.tell() // (50 * 2**20) != (f.tell() - len(chunk)) // (50 * 2**20):
                speed = (f.tell() - done) / (time.time() - t0) / 2**20
                print(f"[{name}] {f.tell()/2**20:.0f}MB / {total/2**20:.0f}MB "
                      f"({100*f.tell()/total:.0f}%) {speed:.1f}MB/s", flush=True)

    got = zip_path.stat().st_size
    if got != total:
        raise SystemExit(f"[{name}] 体积不符 / size mismatch: {got} != {total}（重新运行可续传）")
    print(f"[{name}] 下载完成并校验通过 / verified ({got/2**20:.0f}MB, "
          f"耗时 {(time.time()-t0)/60:.1f} 分钟)")
    check_c_free("下载后 / after")


def extract(name: str, cfg: dict):
    zip_path, dest = cfg["zip"], cfg["extract_to"]
    print(f"[{name}] 解压到 / extracting -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist()
                   if not any(j in m.filename for j in JUNK)
                   and not m.filename.endswith((".db", ".ini"))]
        z.extractall(dest, members=members)
    print(f"[{name}] 解压完成 / done: {len(members)} 个文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mrl", "columbia", "all"], default="all")
    ap.add_argument("--extract", action="store_true", help="下载后解压 / extract after download")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, DATASETS[name])
        if args.extract:
            extract(name, DATASETS[name])


if __name__ == "__main__":
    main()
