#!/usr/bin/env python
"""三范式 AWBC 数据准备 (awbc_three_paradigm_comparison_plan §2/§4).

从已建 A_v4_chunk001_dagger_crave_human 派生两个数据集(仅改 parquet 列 + tasks.jsonl, 视频 symlink 复用):

  1. _human  (A1 COND):   task_index = intervention {0=neg人控外, 1=pos人控}, tasks.jsonl 2 条 advantage prompt
  2. _cls    (A2/A3):      task_index = 0 (中性单任务), 新增 sample_weight 列 = classmap, tasks.jsonl 1 条中性

classmap (dagger_frame_class → sample_weight, plan §2):
  5=base:1.0   0=robot:1.0   1=intv(抓取):2.0   2=preintv:0.0

用法:
  python enrich_chunk001_three_paradigm.py --src <_human 源> --out-root <self_built 目录>
  --link-videos 时视频用软链(本地验证省空间); North-E 上跑时视频已由 build 生成, 用 --videos-from 指真实源。
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CLASSMAP = {5: 1.0, 0: 1.0, 1: 2.0, 2: 0.0}   # dagger_frame_class → sample_weight (A2 loss / A3 excl)
# A3: task_index = 重映射 class → 连续 domain, 走现成 _DomainWeightedJAXSampler (零新代码)。
#   robot0→0(w1)  intv1→1(w2)  preintv2→2(w0)  base5→3(w1)
CLASS_REMAP = {0: 0, 1: 1, 2: 2, 5: 3}
DOMAIN_WEIGHTS = {0: 1, 1: 2, 2: 0, 3: 1}      # A3 config 里的 domain_sample_weights (供记录)
NEG_PROMPT = "Flatten and fold the cloth. Advantage: negative"
POS_PROMPT = "Flatten and fold the cloth. Advantage: positive"
NEUTRAL = "Flatten and fold the cloth."


def _link_or_copy_tree(src: Path, dst: Path, link: bool):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            if link:
                target.symlink_to(p.resolve())
            else:
                shutil.copy2(p, target)


def _write_meta(src_meta: Path, dst_meta: Path, tasks: list[str]):
    dst_meta.mkdir(parents=True, exist_ok=True)
    # copy episodes / episodes_stats / info verbatim
    for fn in ("episodes.jsonl", "episodes_stats.jsonl", "info.json"):
        if (src_meta / fn).exists():
            shutil.copy2(src_meta / fn, dst_meta / fn)
    with open(dst_meta / "tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")


def build_variant(src: Path, out: Path, *, mode: str, link_videos: bool):
    print(f"\n=== build {mode} → {out.name} ===")
    if out.exists():
        shutil.rmtree(out)
    # videos: symlink (省空间); 真实 North-E 构建时视频已在, 此处仅复用 src 的
    _link_or_copy_tree(src / "videos", out / "videos", link=link_videos)

    pq_files = sorted((src / "data").rglob("*.parquet"))
    (out / "data").mkdir(parents=True, exist_ok=True)
    tot = 0
    n_pos = n_neg = 0
    w_hist = {}
    for f in pq_files:
        t = pq.read_table(f)
        cols = t.column_names
        cls = np.asarray(t["dagger_frame_class"]) if "dagger_frame_class" in cols else None
        iv = np.asarray(t["intervention"]) if "intervention" in cols else None
        n = t.num_rows
        d = {c: t[c] for c in cols}

        if mode == "human":
            # task_index = intervention (1=pos人控={base,intv}, 0=neg={robot,preintv})
            assert iv is not None, f"{f} missing intervention"
            ti = iv.astype(np.int64)
            d["task_index"] = pa.array(ti)
            n_pos += int((ti == 1).sum()); n_neg += int((ti == 0).sum())
        elif mode == "cls":
            # task_index = 重映射 class (A3 sampler 的 domain); sample_weight 列 (A2 loss)
            assert cls is not None, f"{f} missing dagger_frame_class"
            ti = np.array([CLASS_REMAP.get(int(c), 0) for c in cls], dtype=np.int64)
            sw = np.array([CLASSMAP.get(int(c), 1.0) for c in cls], dtype=np.float32)
            d["task_index"] = pa.array(ti)
            d["sample_weight"] = pa.array(sw)
            for c in np.unique(cls):
                wv = float(CLASSMAP.get(int(c), 1.0))
                w_hist[wv] = w_hist.get(wv, 0) + int((cls == c).sum())
        out_t = pa.table(d)
        rel = f.relative_to(src / "data")
        (out / "data" / rel.parent).mkdir(parents=True, exist_ok=True)
        pq.write_table(out_t, out / "data" / rel)
        tot += n

    if mode == "human":
        _write_meta(src / "meta", out / "meta", [NEG_PROMPT, POS_PROMPT])
        print(f"  frames={tot}  positive={n_pos} ({n_pos/tot:.1%})  negative={n_neg} ({n_neg/tot:.1%})")
        print(f"  tasks.jsonl: 0={NEG_PROMPT!r}  1={POS_PROMPT!r}")
    else:
        # 4 条 task 全中性 (task_index 0..3 = 重映射 class); A2/A3 都用中性 prompt
        _write_meta(src / "meta", out / "meta", [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL])
        print(f"  frames={tot}  sample_weight 分布: {dict(sorted(w_hist.items()))}")
        print(f"  task_index(重映射class)分布已写; A3 domain_sample_weights={DOMAIN_WEIGHTS}")
        print(f"  tasks.jsonl: 4×{NEUTRAL!r}  (+ sample_weight 列)")
    # copy norm_stats
    for fn in ("norm_stats.json",):
        if (src / fn).exists():
            shutil.copy2(src / fn, out / fn)
    print(f"  ✅ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="A_v4_chunk001_dagger_crave_human 源目录")
    ap.add_argument("--out-root", required=True, help="self_built 输出根目录")
    ap.add_argument("--link-videos", action="store_true", help="视频软链(本地验证省空间)")
    ap.add_argument("--only", choices=["human", "cls"], help="只建其一")
    a = ap.parse_args()
    src = Path(a.src); root = Path(a.out_root)
    assert (src / "meta/episodes.jsonl").exists(), f"src 无效: {src}"

    if a.only in (None, "human"):
        build_variant(src, root / "A_v4_chunk001_3para_human", mode="human", link_videos=a.link_videos)
    if a.only in (None, "cls"):
        build_variant(src, root / "A_v4_chunk001_3para_cls", mode="cls", link_videos=a.link_videos)
    print("\n[done]")


if __name__ == "__main__":
    main()
