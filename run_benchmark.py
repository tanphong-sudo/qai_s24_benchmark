#!/usr/bin/env python3
"""Qualcomm AI Hub hosted-S24 ASR benchmark.
The laptop only orchestrates preprocessing/API calls; neural encoder/decoder inference runs on the hosted Samsung Galaxy S24 NPU.
"""

import os, sys, re, gc, io, json, math, time, random, hashlib, shutil, zipfile, unicodedata, platform, inspect
from datetime import datetime, timezone
import importlib.metadata as im
from dataclasses import dataclass
from pathlib import Path
from getpass import getpass
from urllib.request import urlopen
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm

import torch
import qai_hub as hub
from datasets import Audio, load_dataset
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from transformers import AutoProcessor, GenerationConfig, WhisperConfig
from jiwer import process_words, process_characters

try:
    from IPython.display import display
except Exception:
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string(index=False))
        else:
            print(obj)

from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisper,
    HfWhisperEncoder,
    HfWhisperDecoder,
    MEAN_DECODE_LEN,
    SAMPLE_RATE,
)

PATCH_VERSION = "2026-08-21-torchcodec-free-audio-v11"
SEED = 42
# Default is the requested fixed-size benchmark, NOT the huge full split.
RUN_MODE = os.environ.get("QAI_RUN_MODE", "benchmark").strip().lower()
BENCHMARK_N = 50             # 50 samples for each benchmark category
SMOKE_N = 3
SMOKE_VIMD_PER_REGION = 2

# ViMD still exposes all 3 regional columns while keeping 50 utterances total.
# 17 + 17 + 16 = 50.
VIMD_BENCHMARK_REGION_TARGETS = {"North": 17, "Central": 17, "South": 16}

# Code-switch table uses the official ViMedCSS test split. Set True only if you
# explicitly also want another 50 from the harder diagnostic split.
VIMEDCSS_INCLUDE_HARD = False

# Pack independent utterances into one Workbench job. Colab can request a
# larger batch; adaptive splitting preserves the same predictions if it fails.
HUB_MICROBATCH = max(1, int(os.environ.get("QAI_HUB_MICROBATCH", "2")))
HUB_JOB_RETRIES = max(1, int(os.environ.get("QAI_HUB_JOB_RETRIES", "3")))
ARTIFACT_BUILD_POLICY = os.environ.get("QAI_ARTIFACT_POLICY", "separate_qnn_dlc").strip().lower()
ENABLE_PROFILING = os.environ.get("QAI_ENABLE_PROFILING", "1").strip().lower() in {"1", "true", "yes"}
QAIRT_VERSION_REQUEST = "latest"
TARGET_DEVICE_NAME = "Samsung Galaxy S24"  # exact hosted device, NEVER Family/Proxy fallback
SR = 16_000
MAX_AUDIO_SECONDS = 30.0
TARGET_SNR_DB = 0.0

RUN = {
    "fleurs_vi_clean": True,
    "fleurs_vi_noise_0db": True,
    "fleurs_en_clean": True,
    "vimd_regional": True,
    "vimedcss_codeswitch": True,
}

MODEL_IDS = {
    "Whisper Tiny": "openai/whisper-tiny",
    "Whisper Small": "openai/whisper-small",
    "PhoWhisper Base": "vinai/PhoWhisper-base",
}

IS_COLAB = "google.colab" in sys.modules
USE_GOOGLE_DRIVE = IS_COLAB
if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        WORK_ROOT = Path("/content/drive/MyDrive/qai_asr_s24_benchmark")
    except Exception as exc:
        print("Google Drive mount unavailable; using /content. Reason:", repr(exc))
        WORK_ROOT = Path("/content/qai_asr_s24_benchmark")
else:
    # Laptop: override with QAI_BENCHMARK_ROOT if desired.
    WORK_ROOT = Path(os.environ.get("QAI_BENCHMARK_ROOT", str(Path.cwd() / "qai_asr_s24_benchmark"))).expanduser().resolve()

HF_HOME = Path(os.environ.get("QAI_HF_HOME", str(WORK_ROOT / "hf_cache"))).expanduser().resolve()
ARTIFACT_DIR = WORK_ROOT / "qualcomm_artifacts"
CHECKPOINT_DIR = WORK_ROOT / "checkpoints"
RESULT_DIR = WORK_ROOT / "results" / RUN_MODE
DATA_DIR = Path(os.environ.get("QAI_DATA_ROOT", str(WORK_ROOT / "data"))).expanduser().resolve()
for p in [HF_HOME, ARTIFACT_DIR, CHECKPOINT_DIR, RESULT_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)

RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
RUN_CONSOLE_LOG = WORK_ROOT / "RUN_CONSOLE.log"
class _TeeStream:
    def __init__(self, primary, log_path):
        self.primary = primary
        self.log = open(log_path, "a", encoding="utf-8", buffering=1)
    def write(self, data):
        self.primary.write(data)
        self.log.write(data)
        self.log.flush()
        return len(data)
    def flush(self):
        self.primary.flush()
        self.log.flush()
    def isatty(self):
        return bool(getattr(self.primary, "isatty", lambda: False)())
    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")

# Keep an audit log locally. The API token is entered with getpass and is never printed/stored.
sys.stdout = _TeeStream(sys.stdout, RUN_CONSOLE_LOG)
sys.stderr = _TeeStream(sys.stderr, RUN_CONSOLE_LOG)

JOB_LEDGER_JSONL = RESULT_DIR / "QUALCOMM_JOB_LEDGER.jsonl"
def append_job_evidence(record: dict):
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with open(JOB_LEDGER_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("run_benchmark patch:", PATCH_VERSION)
print({
    "run_mode": RUN_MODE,
    "benchmark_n": BENCHMARK_N,
    "work_root": str(WORK_ROOT),
    "hub_microbatch": HUB_MICROBATCH,
    "hub_job_retries": HUB_JOB_RETRIES,
    "artifact_build_policy": ARTIFACT_BUILD_POLICY,
    "profiling_enabled": ENABLE_PROFILING,
    "max_decode_len": MEAN_DECODE_LEN,
    "target_device": TARGET_DEVICE_NAME,
    "qairt_requested": QAIRT_VERSION_REQUEST,
})
print("Python", sys.version.split()[0], "| torch", torch.__version__, "| qai-hub", im.version("qai-hub"),
      "| qai-hub-models", im.version("qai-hub-models"))
assert RUN_MODE in {"smoke", "benchmark", "full"}
assert HUB_MICROBATCH >= 1
assert ARTIFACT_BUILD_POLICY in {"separate_qnn_dlc", "linked_context"}
assert sum(VIMD_BENCHMARK_REGION_TARGETS.values()) == BENCHMARK_N
assert MEAN_DECODE_LEN == 200, "This benchmark's decoder state layout assumes Qualcomm HfWhisper MEAN_DECODE_LEN=200."


def _colab_secret(name: str):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None

api_token = os.environ.get("QAI_HUB_API_TOKEN") or _colab_secret("QAI_HUB_API_TOKEN")
if not api_token:
    api_token = getpass("Qualcomm AI Hub API token: ").strip()
if not api_token:
    raise RuntimeError("Missing Qualcomm AI Hub API token")

# Configure Qualcomm AI Hub directly through the Python API.
# This is Qualcomm's documented "session API token" flow and avoids relying
# on the `qai-hub` command being present on Windows PATH.
client_config = hub.ClientConfig(api_token=api_token)
client = hub.Client(client_config)
del api_token
print("Qualcomm AI Hub session authenticated via Python ClientConfig.")
if not hasattr(client, "submit_compile_job"):
    raise RuntimeError("Installed qai-hub client is too old: submit_compile_job() is required.")
if ARTIFACT_BUILD_POLICY=="linked_context" and not hasattr(client, "submit_compile_and_link_jobs"):
    raise RuntimeError("QAI_ARTIFACT_POLICY=linked_context requires submit_compile_and_link_jobs().")

exact_devices = client.get_devices(name=TARGET_DEVICE_NAME)
if not exact_devices:
    raise RuntimeError(
        f"Your Workbench account exposes no exact '{TARGET_DEVICE_NAME}'. "
        "This notebook deliberately will NOT fall back to an S24 Family or Proxy device."
    )
TARGET_DEVICE = exact_devices[0]
if TARGET_DEVICE.name != TARGET_DEVICE_NAME or "(Family)" in TARGET_DEVICE.name or "(Proxy)" in TARGET_DEVICE.name:
    raise AssertionError(f"Not an exact hosted S24: {TARGET_DEVICE}")

# Freeze the mutable QAIRT tag ("latest"/"default") to one exact API version for this run.
WORKBENCH_FRAMEWORKS = sorted([
    {
        "name": str(getattr(f, "name", "")),
        "api_version": str(getattr(f, "api_version", "")),
        "full_version": str(getattr(f, "full_version", "")),
        "api_tags": sorted(map(str, getattr(f, "api_tags", []) or [])),
    }
    for f in client.get_frameworks()
], key=lambda x: (x["name"], x["api_version"], x["full_version"]))
qairt = [x for x in WORKBENCH_FRAMEWORKS if x["name"].upper() == "QAIRT"]
if QAIRT_VERSION_REQUEST in {"latest", "default"}:
    matches = [x for x in qairt if QAIRT_VERSION_REQUEST in x["api_tags"]]
else:
    matches = [x for x in qairt if x["api_version"] == QAIRT_VERSION_REQUEST]
if len(matches) != 1:
    raise RuntimeError(
        f"Could not uniquely resolve QAIRT {QAIRT_VERSION_REQUEST!r}; available QAIRT entries: {qairt}"
    )
QAIRT_VERSION = matches[0]["api_version"]
QAIRT_FULL_VERSION = matches[0]["full_version"]

DEVICE_FINGERPRINT_PAYLOAD = {
    "name": TARGET_DEVICE.name,
    "os": str(getattr(TARGET_DEVICE, "os", "")),
    "attributes": sorted(map(str, getattr(TARGET_DEVICE, "attributes", []) or [])),
}
DEVICE_FINGERPRINT = hashlib.sha256(
    json.dumps(DEVICE_FINGERPRINT_PAYLOAD, sort_keys=True).encode()
).hexdigest()
WORKBENCH_FRAMEWORK_FINGERPRINT = hashlib.sha256(
    json.dumps(WORKBENCH_FRAMEWORKS, sort_keys=True).encode()
).hexdigest()

print("Exact hosted target:", TARGET_DEVICE)
print("Attributes:", getattr(TARGET_DEVICE, "attributes", None))
print(f"QAIRT {QAIRT_VERSION_REQUEST!r} resolved to API {QAIRT_VERSION} / full {QAIRT_FULL_VERSION}")


hf = HfApi()
MODEL_REVISIONS = {repo: hf.model_info(repo).sha for repo in MODEL_IDS.values()}
DATASET_REVISIONS = {
    "google/fleurs": hf.dataset_info("google/fleurs").sha,
    "nguyendv02/ViMD_Dataset": hf.dataset_info("nguyendv02/ViMD_Dataset").sha,
    "tensorxt/ViMedCSS": hf.dataset_info("tensorxt/ViMedCSS").sha,
}

BASE_MANIFEST = {
    "seed": SEED,
    "target_device": TARGET_DEVICE_NAME,
    "qairt_requested": QAIRT_VERSION_REQUEST,
    "qairt_api_version": QAIRT_VERSION,
    "qairt_full_version": QAIRT_FULL_VERSION,
    "workbench_framework_fingerprint": WORKBENCH_FRAMEWORK_FINGERPRINT,
    "workbench_frameworks": WORKBENCH_FRAMEWORKS,
    "device_fingerprint": DEVICE_FINGERPRINT,
    "device_fingerprint_payload": DEVICE_FINGERPRINT_PAYLOAD,
    "qai_hub_version": im.version("qai-hub"),
    "qai_hub_models_version": im.version("qai-hub-models"),
    "torch_version": torch.__version__,
    "models": MODEL_IDS,
    "model_revisions": MODEL_REVISIONS,
    "dataset_revisions": DATASET_REVISIONS,
    "sample_rate": SR,
    "max_audio_seconds": MAX_AUDIO_SECONDS,
    "run_mode": RUN_MODE,
    "benchmark_n": BENCHMARK_N,
    "vimd_benchmark_region_targets": VIMD_BENCHMARK_REGION_TARGETS,
    "vimedcss_include_hard": VIMEDCSS_INCLUDE_HARD,
    "hub_microbatch": HUB_MICROBATCH,
    "hub_job_retries": HUB_JOB_RETRIES,
    "artifact_build_policy": ARTIFACT_BUILD_POLICY,
    "profiling_enabled": ENABLE_PROFILING,
    "mean_decode_len": MEAN_DECODE_LEN,
    "decoding": {
        "greedy": True,
        "do_sample": False,
        "num_beams": 1,
        "task": "transcribe",
        "timestamps": False,
        "condition_on_previous_text": False,
    },
}
(RESULT_DIR / "run_manifest.json").write_text(json.dumps({**BASE_MANIFEST, "run_mode": RUN_MODE}, indent=2, ensure_ascii=False))
print(json.dumps(BASE_MANIFEST, indent=2, ensure_ascii=False))


CONTROL_TOKEN_RE = re.compile(r"<\|[^<>]*\|>|</?s>|<pad>|<unk>", flags=re.IGNORECASE)

def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = CONTROL_TOKEN_RE.sub(" ", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())

def normalize_en(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = CONTROL_TOKEN_RE.sub(" ", text).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.split())

assert normalize_vi("Xin CHÀO, bạn!") == "xin chào bạn"
assert normalize_vi("<|vi|><|transcribe|> Mở van") == "mở van"
assert normalize_vi("mo van") != "mở van"
assert normalize_en("Hello, WORLD!") == "hello world"
assert normalize_en("don't") != "do not"

def word_counts(ref: str, hyp: str) -> dict[str, Any]:
    m = process_words(ref, hyp)
    n = m.hits + m.substitutions + m.deletions
    return dict(hits=m.hits, substitutions=m.substitutions, deletions=m.deletions,
                insertions=m.insertions, reference_words=n, sample_wer=m.wer)

def char_counts(ref: str, hyp: str) -> dict[str, Any]:
    m = process_characters(ref, hyp)
    n = m.hits + m.substitutions + m.deletions
    return dict(char_hits=m.hits, char_substitutions=m.substitutions, char_deletions=m.deletions,
                char_insertions=m.insertions, reference_chars=n, sample_cer=m.cer)

def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _json_default(obj: Any):
    # Qualcomm qai-hub-models input specs contain TensorSpec objects, which are
    # runtime objects and are not JSON-serializable by Python's default encoder.
    # The manifest only needs a stable audit/cache representation; the real
    # TensorSpec objects are reconstructed from the pinned model on reuse.
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)

    if obj.__class__.__name__ == "TensorSpec":
        value = {"__type__": "TensorSpec"}
        for attr in ("name", "shape", "dtype"):
            if hasattr(obj, attr):
                v = getattr(obj, attr)
                if attr == "shape":
                    try:
                        v = list(v)
                    except TypeError:
                        v = str(v)
                elif attr == "dtype":
                    v = str(v)
                value[attr] = v
        value["repr"] = repr(obj)
        return value

    # Evidence/cache JSON must never abort a successful Qualcomm job merely
    # because an SDK object gained a non-JSON field in a newer release.
    return repr(obj)

def atomic_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    )
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)

def corpus_summary(detailed: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out=[]
    grouped = detailed.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), detailed)]
    for key,g in grouped:
        if not isinstance(key, tuple): key=(key,)
        r={c:v for c,v in zip(group_cols,key)}
        S,D,I,N = map(int, [g.substitutions.sum(), g.deletions.sum(), g.insertions.sum(), g.reference_words.sum()])
        cS,cD,cI,cN = map(int, [g.char_substitutions.sum(), g.char_deletions.sum(), g.char_insertions.sum(), g.reference_chars.sum()])
        r.update({
            "samples": len(g), "audio_hours": float(g.duration_sec.sum()/3600),
            "reference_words": N, "S": S, "D": D, "I": I,
            "wer": (S+D+I)/N if N else np.nan, "wer_percent": 100*(S+D+I)/N if N else np.nan,
            "cer": (cS+cD+cI)/cN if cN else np.nan, "cer_percent": 100*(cS+cD+cI)/cN if cN else np.nan,
            "mean_utterance_wer_percent": float(100*g.sample_wer.mean()),
            "empty_predictions": int((g.normalized_prediction == "").sum()),
            "samples_wer_over_100_percent": int((g.sample_wer > 1).sum()),
            "decode_steps": int(g.decode_steps.sum()),
            "api_wall_hours": float(g.api_wall_s_share.sum()/3600),
            "estimated_device_compute_sec": (float(g.estimated_device_compute_ms.sum(min_count=1)/1000)
                                             if pd.notna(g.estimated_device_compute_ms.sum(min_count=1)) else np.nan),
        })
        dur=float(g.duration_sec.sum())
        _dev_sum=g.estimated_device_compute_ms.sum(min_count=1)
        dev=float(_dev_sum/1000) if pd.notna(_dev_sum) else np.nan
        r["estimated_device_rtf"] = dev/dur if dur and np.isfinite(dev) else np.nan
        r["estimated_device_x_realtime"] = dur/dev if np.isfinite(dev) and dev>0 else np.nan
        out.append(r)
    return pd.DataFrame(out)



def pairwise_fairness(detailed: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    expected=list(MODEL_IDS)
    x=(detailed.assign(evaluated=True)
       .pivot_table(index="sample_key",columns="model",values="evaluated",aggfunc="max",fill_value=False)
       .reindex(columns=expected,fill_value=False))
    if len(x)!=len(metadata) or not x.to_numpy(dtype=bool).all():
        raise AssertionError(
            f"Pairwise fairness failed: expected {len(metadata)} identical samples for {expected}, got {x.shape}"
        )
    x.columns=[f"{c} evaluated?" for c in x.columns]
    return x.reset_index()

def normalization_audit(metadata: pd.DataFrame, detailed: pd.DataFrame) -> pd.DataFrame:
    base_cols=[c for c in ["sample_key","dataset_index","sample_id","raw_reference","normalized_reference"] if c in metadata.columns]
    base=metadata[base_cols].drop_duplicates("sample_key").copy()
    raw=detailed.pivot(index="sample_key",columns="model",values="raw_prediction").add_prefix("raw_prediction__").reset_index()
    norm=detailed.pivot(index="sample_key",columns="model",values="normalized_prediction").add_prefix("normalized_prediction__").reset_index()
    return base.merge(raw,on="sample_key",how="left").merge(norm,on="sample_key",how="left")

def top_error_audit(detailed: pd.DataFrame, n: int=10) -> pd.DataFrame:
    keep=[c for c in ["sample_key","dataset_index","sample_id","region","province","condition",
                       "raw_reference","normalized_reference","raw_prediction","normalized_prediction",
                       "substitutions","deletions","insertions","reference_words","sample_wer"] if c in detailed.columns]
    frames=[]
    for model in MODEL_IDS:
        g=detailed[detailed.model==model].nlargest(n,"sample_wer")[keep].copy()
        g.insert(0,"audit_rank",np.arange(1,len(g)+1));g.insert(0,"model",model)
        g["sample_wer_percent"]=100*g["sample_wer"]
        frames.append(g)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def print_wer_ranking(summary: pd.DataFrame, title: str):
    if summary.empty:return
    ranked=summary.sort_values("wer").reset_index(drop=True)
    print("\n"+title)
    for i,row in ranked.iterrows():
        print(f"{i+1}. {row['model']}: corpus WER={row['wer_percent']:.2f}%")
    if len(ranked)>1:
        winner=ranked.iloc[0]
        print("Relative WER reduction of winner:")
        for _,other in ranked.iloc[1:].iterrows():
            red=(other.wer-winner.wer)/other.wer if other.wer else np.nan
            print(f"  vs {other['model']}: {100*red:.2f}%")

# --- ViMedCSS CS annotation alignment ---
def parse_cs_terms(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return []
    return [normalize_vi(x) for x in str(raw).split(";") if normalize_vi(x)]

def align_cs_terms_to_reference(normalized_ref: str, raw_terms: Any):
    ref = normalized_ref.split()
    mask = [False]*len(ref)
    failures=[]
    for term in parse_cs_terms(raw_terms):
        tw=term.split(); found=None
        for start in range(0, len(ref)-len(tw)+1):
            if any(mask[start:start+len(tw)]): continue
            if ref[start:start+len(tw)] == tw:
                found=start; break
        if found is None:
            failures.append(term)
        else:
            for i in range(found, found+len(tw)): mask[i]=True
    return mask, failures

def levenshtein_ops(ref_tokens: list[str], hyp_tokens: list[str]):
    n,m=len(ref_tokens),len(hyp_tokens)
    dp=np.zeros((n+1,m+1),dtype=np.int32)
    dp[:,0]=np.arange(n+1); dp[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=0 if ref_tokens[i-1]==hyp_tokens[j-1] else 1
            dp[i,j]=min(dp[i-1,j-1]+cost, dp[i-1,j]+1, dp[i,j-1]+1)
    ops=[]; i=n; j=m
    while i or j:
        if i and j:
            cost=0 if ref_tokens[i-1]==hyp_tokens[j-1] else 1
            if dp[i,j] == dp[i-1,j-1]+cost:
                ops.append(("hit" if cost==0 else "sub", i-1, j-1, i-1))
                i-=1; j-=1; continue
        if i and dp[i,j] == dp[i-1,j]+1:
            ops.append(("del", i-1, None, i-1)); i-=1; continue
        if j and dp[i,j] == dp[i,j-1]+1:
            # insertion at reference boundary i: between ref[i-1] and ref[i]
            ops.append(("ins", None, j-1, i)); j-=1; continue
        raise AssertionError("Levenshtein backtrace failed")
    return list(reversed(ops))

def projected_cs_metrics(normalized_ref: str, normalized_hyp: str, cs_mask: list[bool]):
    ref=normalized_ref.split(); hyp=normalized_hyp.split()
    if len(ref)!=len(cs_mask): raise ValueError("CS mask length mismatch")
    counters={"cs":{"H":0,"S":0,"D":0,"I":0,"N":int(sum(cs_mask))},
              "n":{"H":0,"S":0,"D":0,"I":0,"N":int(len(ref)-sum(cs_mask))}}
    for op,ri,hj,boundary in levenshtein_ops(ref,hyp):
        if op in {"hit","sub","del"}:
            cls="cs" if cs_mask[ri] else "n"
            counters[cls][{"hit":"H","sub":"S","del":"D"}[op]] += 1
        else:
            left_cs = boundary>0 and cs_mask[boundary-1]
            right_cs = boundary<len(cs_mask) and cs_mask[boundary]
            cls = "cs" if (left_cs or right_cs) else "n"
            counters[cls]["I"] += 1
    out={}
    for cls,prefix in [("cs","cs"),("n","n")]:
        c=counters[cls]; N=c["N"]
        out.update({f"{prefix}_hits":c["H"],f"{prefix}_S":c["S"],f"{prefix}_D":c["D"],f"{prefix}_I":c["I"],f"{prefix}_N":N,
                    f"{prefix}_wer":(c["S"]+c["D"]+c["I"])/N if N else np.nan,
                    f"{prefix}_token_accuracy":c["H"]/N if N else np.nan})
    return out

# unit tests
m = projected_cs_metrics("uống aspirin mỗi ngày", "uống asprin mỗi ngày", [False,True,False,False])
assert m["cs_S"]==1 and m["cs_N"]==1 and abs(m["cs_wer"]-1.0)<1e-12
print("Normalization, WER/CER, and CS projection tests passed.")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def model_cache_key(label: str, repo: str) -> str:
    payload={
        "label":label,"repo":repo,"revision":MODEL_REVISIONS[repo],
        "device":TARGET_DEVICE_NAME,"device_fingerprint":DEVICE_FINGERPRINT,
        "qairt":QAIRT_VERSION,"qairt_full":QAIRT_FULL_VERSION,
        "workbench_framework_fingerprint":WORKBENCH_FRAMEWORK_FINGERPRINT,
        "qhm":im.version("qai-hub-models"),"torch":torch.__version__,
        "mean_decode_len":MEAN_DECODE_LEN,"compile_policy":"float16_quantize_io",
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def source_snapshot(repo: str) -> Path:
    rev=MODEL_REVISIONS[repo]
    local=HF_HOME / "model_snapshots" / slug(repo) / rev
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=rev, local_dir=str(local))
    return local

def _call_qhm_compat(bound_method, *legacy_args):
    """Call a qai-hub-models helper across old/new signatures.

    In qai-hub-models 0.57.x, HfWhisperEncoder/Decoder.get_input_spec() is an
    instance method with no explicit shape arguments. Older snippets passed
    config values. We inspect the *bound* signature so this notebook works with
    the pinned 0.57.0 API and remains tolerant of the older signature.
    """
    sig = inspect.signature(bound_method)
    positional = [p for p in sig.parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        return bound_method(*legacy_args)
    if len(positional) > len(legacy_args):
        raise TypeError(f"Cannot satisfy {bound_method} signature {sig} with {len(legacy_args)} compatibility args")
    return bound_method(*legacy_args[:len(positional)])

def _component_contracts(original):
    config=original.config
    encoder=HfWhisperEncoder(config, original.get_encoder())
    decoder=HfWhisperDecoder(config, original.get_decoder())
    enc_spec=_call_qhm_compat(encoder.get_input_spec, config.num_mel_bins)
    dec_spec=_call_qhm_compat(decoder.get_input_spec, config.decoder_layers, config.d_model, config.decoder_attention_heads)
    enc_out=_call_qhm_compat(encoder.get_output_names, config.decoder_layers)
    dec_out=_call_qhm_compat(decoder.get_output_names, config.decoder_layers)
    return config, encoder, decoder, enc_spec, dec_spec, enc_out, dec_out

def export_whisper_components(label: str, repo: str):
    print(f"Loading pinned source for {label}: {repo}@{MODEL_REVISIONS[repo]}")
    snap=source_snapshot(repo)
    original=HfWhisper.load_whisper_model(str(snap))
    config,encoder,decoder,enc_spec,dec_spec,enc_out,dec_out=_component_contracts(original)

    local_root = Path("/content/qai_source_models") if IS_COLAB else (WORK_ROOT / "qai_source_models")
    local_dir=local_root/slug(label)
    if local_dir.exists(): shutil.rmtree(local_dir)
    (local_dir/"encoder").mkdir(parents=True, exist_ok=True)
    (local_dir/"decoder").mkdir(parents=True, exist_ok=True)
    # BaseModel.serialize uses Qualcomm's tested torch.jit.trace path and exact input order.
    enc_path=encoder.serialize(local_dir/"encoder", enc_spec)
    dec_path=decoder.serialize(local_dir/"decoder", dec_spec)

    meta={
        "snapshot":str(snap), "config":config.to_dict(),
        "encoder_input_spec":enc_spec,"decoder_input_spec":dec_spec,
        "encoder_output_names":enc_out,"decoder_output_names":dec_out,
        "encoder_path":str(enc_path),"decoder_path":str(dec_path),
    }
    del encoder,decoder,original
    gc.collect()
    return meta

@dataclass(frozen=True)
class _LocalTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

def _normalize_compile_input_spec(spec: Any) -> list[_LocalTensorSpec]:
    """Normalize qai-hub-models compile InputSpec into TensorSpec-like rows.

    Qualcomm Workbench normally exposes Model.input_spec for QNN DLC/context
    artifacts, but some client/server combinations return an empty IOSpec. The
    exact input spec submitted to the compile job remains authoritative.
    """
    out=[]
    if isinstance(spec, dict):
        items=list(spec.items())
    elif isinstance(spec, (list, tuple)):
        items=[(getattr(v, "name", None), v) for v in spec]
    else:
        raise TypeError(f"Unsupported compile input spec type: {type(spec)}")
    for key, value in items:
        name=str(getattr(value, "name", None) or key)
        if hasattr(value, "shape"):
            shape=tuple(int(x) for x in value.shape)
            dtype=str(getattr(value, "dtype", "float32"))
        elif isinstance(value, (tuple, list)) and len(value)==2 and isinstance(value[0], (tuple, list)) and isinstance(value[1], str):
            shape=tuple(int(x) for x in value[0]); dtype=str(value[1])
        else:
            shape=tuple(int(x) for x in value); dtype="float32"
        out.append(_LocalTensorSpec(name=name, shape=shape, dtype=dtype))
    return out

def inspect_artifact_contract(encoder_model, decoder_model, cached: dict):
    graph_names=cached["graph_names"]
    rows=[]
    used_remote_inputs=False
    used_remote_outputs=False
    for idx,(graph,model) in enumerate(zip(graph_names,[encoder_model,decoder_model])):
        remote_inputs=getattr(model, "input_spec", {}) or {}
        remote_outputs=getattr(model, "output_spec", {}) or {}
        local_input = cached["encoder_input_spec"] if idx==0 else cached["decoder_input_spec"]
        local_outputs = cached["encoder_output_names"] if idx==0 else cached["decoder_output_names"]
        remote_in_specs=remote_inputs.get(graph)
        if not remote_in_specs and len(remote_inputs)==1 and None in remote_inputs:
            remote_in_specs=remote_inputs[None]
        in_specs=remote_in_specs or _normalize_compile_input_spec(local_input)
        used_remote_inputs=used_remote_inputs or bool(remote_in_specs)
        for t in in_specs:
            rows.append({"graph":graph,"side":"input","name":t.name,"shape":tuple(t.shape),"dtype":str(t.dtype),
                         "contract_source":"workbench_model_spec" if remote_in_specs else "submitted_compile_spec"})
        out_specs=remote_outputs.get(graph)
        if not out_specs and len(remote_outputs)==1 and None in remote_outputs:
            out_specs=remote_outputs[None]
        out_specs=out_specs or []
        used_remote_outputs=used_remote_outputs or bool(out_specs)
        if out_specs:
            for t in out_specs:
                rows.append({"graph":graph,"side":"output","name":t.name,"shape":tuple(t.shape),"dtype":str(t.dtype),
                             "contract_source":"workbench_model_spec"})
        else:
            for name in local_outputs:
                rows.append({"graph":graph,"side":"output","name":str(name),"shape":None,"dtype":None,
                             "contract_source":"submitted_output_names"})
    if not used_remote_inputs:
        print("NOTE: Workbench returned no usable artifact input_spec; using the exact input specs submitted at compile time for local validation.")
    if not used_remote_outputs:
        print("NOTE: Workbench returned no usable artifact output_spec; using the exact --output_names submitted at compile time.")
    return pd.DataFrame(rows)

REMOTE_MANIFEST_PATH=ARTIFACT_DIR/"hub_artifacts.json"

def _safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        bad=path.with_suffix(path.suffix+f".corrupt_{int(time.time())}")
        try: path.replace(bad)
        except Exception: pass
        print(f"WARNING: invalid JSON cache {path}; starting fresh. Reason: {exc!r}")
        return default

def _status_line(status) -> str:
    if status is None:
        return "<no status>"
    return f"code={getattr(status,'code',None)!r}, success={getattr(status,'success',False)!r}, failure={getattr(status,'failure',False)!r}, message={getattr(status,'message',None)!r}"

def _wait_job_success(job, what: str):
    status=job.wait()
    print(f"{what}: {_status_line(status)} | {getattr(job,'url',None)}")
    if not bool(getattr(status,"success",False)):
        raise RuntimeError(f"{what} FAILED: {_status_line(status)} | {getattr(job,'url',None)}")
    return status

def _run_inference_job_with_retries(client_obj, model, device, inputs, name: str,
                                    options: str, target_device_name: str, max_attempts: int):
    failures=[]
    for attempt in range(1,max_attempts+1):
        job=None
        try:
            job=client_obj.submit_inference_job(
                model=model,device=device,inputs=inputs,name=name,options=options
            )
            if getattr(getattr(job,"device",None),"name",None)!=target_device_name:
                raise RuntimeError(f"Inference job {getattr(job,'url',None)} is not on exact {target_device_name}: {getattr(job,'device',None)}")
            status=job.wait()
            print(f"Inference {name} attempt {attempt}/{max_attempts}: {_status_line(status)} | {getattr(job,'url',None)}")
            if not bool(getattr(status,"success",False)):
                raise RuntimeError(f"remote status {_status_line(status)}")
            raw=job.download_output_data()
            if raw is None or not hasattr(raw,"keys"):
                raise RuntimeError("successful job returned no in-memory output data")
            return job,raw
        except Exception as exc:
            detail=f"attempt {attempt}: job={getattr(job,'url',None)} error={exc!r}"
            failures.append(detail)
            if attempt<max_attempts:
                print(f"WARNING: inference {name} failed; retrying. {detail}")
    raise RuntimeError(f"Inference {name} failed after {max_attempts} attempts. " + " | ".join(failures))

def _model_producer_status(model):
    ok=bool(model.wait())
    producer=model.get_producer()
    if producer is None:
        return ok, None, "model has no producer"
    status=producer.get_status()
    if not bool(getattr(status,"finished",False)):
        status=producer.wait()
    success=ok and bool(getattr(status,"success",False))
    return success, producer, _status_line(status)

def _require_model_ready(model, label: str):
    success, producer, detail=_model_producer_status(model)
    if not success:
        raise RuntimeError(
            f"{label}: target model {getattr(model,'model_id',None)} has a failed producer. "
            f"producer={getattr(producer,'url',None)}; {detail}"
        )
    return producer

def _link_retry_options(qairt_version: str) -> list[str]:
    return [
        f"--qairt_version {qairt_version} --qnn_options default_graph_htp_optimizations=O=2",
        f"--qairt_version {qairt_version} --qnn_options default_graph_htp_optimizations=O=1",
    ]

def _submit_component_compile_jobs(client_obj, source_models, input_specs, output_names,
                                   device, name: str, qairt_version: str):
    common=f"--target_runtime qnn_dlc --quantize_full_type float16 --quantize_io --qairt_version {qairt_version}"
    jobs=[]
    for component,model,spec,names in zip(
        ("encoder","decoder"),source_models,input_specs,output_names
    ):
        options=f"--output_names {','.join(names)} {common}"
        jobs.append(client_obj.submit_compile_job(
            model=model,
            device=device,
            name=f"{name}_{component}",
            input_specs=spec,
            options=options,
        ))
    return jobs

def _retry_link_jobs(client_obj, models, device, name: str, qairt_version: str):
    attempts=[]
    for options in _link_retry_options(qairt_version):
        try:
            job=client_obj.submit_link_job(models,device=device,name=name,options=options)
            job_device=getattr(job,"device",None)
            if job_device is not None and getattr(job_device,"name",None)!=getattr(device,"name",None):
                raise RuntimeError(f"link retry targeted {job_device}; expected {device}")
            status=job.wait()
            success=bool(getattr(status,"success",False))
            attempt={
                "job_id":getattr(job,"job_id",None),
                "job_url":getattr(job,"url",None),
                "options":options,
                "status":_status_line(status),
                "success":success,
            }
            print(f"Link retry ({options}): {_status_line(status)} | {getattr(job,'url',None)}")
            if success:
                model=job.get_target_model()
                ready=model is not None and bool(model.wait())
                attempt["target_model_id"]=getattr(model,"model_id",None)
                if ready:
                    attempts.append(attempt)
                    return job,model,attempts
                attempt["success"]=False
                attempt["target_error"]="successful link returned no ready target model"
            attempts.append(attempt)
        except Exception as exc:
            attempts.append({"options":options,"success":False,"error":repr(exc)})
            print(f"WARNING: link retry submission/execution failed for {options}: {exc!r}")
    return None,None,attempts

def _cached_artifact_model_ids(cached: dict):
    mode=cached.get("artifact_mode")
    if mode=="separate_qnn_dlc":
        return mode,cached.get("encoder_model_id"),cached.get("decoder_model_id")
    linked_id=cached.get("linked_model_id")
    if linked_id:
        return "linked_context",linked_id,linked_id
    return mode,cached.get("encoder_model_id"),cached.get("decoder_model_id")

remote_manifest=_safe_json_load(REMOTE_MANIFEST_PATH,{})
MODEL_ARTIFACTS={}

for label,repo in MODEL_IDS.items():
    print("\n"+"="*100+f"\n{label}")
    key=model_cache_key(label,repo)
    cached=remote_manifest.get(label)
    artifact_mode=None
    encoder_model=None
    decoder_model=None
    if cached and cached.get("cache_key")==key:
        try:
            artifact_mode,encoder_id,decoder_id=_cached_artifact_model_ids(cached)
            if artifact_mode not in {"linked_context","separate_qnn_dlc"} or not encoder_id or not decoder_id:
                raise ValueError("cached artifact layout is incomplete")
            encoder_model=client.get_model(encoder_id)
            decoder_model=encoder_model if decoder_id==encoder_id else client.get_model(decoder_id)
            for component,candidate in [("encoder",encoder_model),("decoder",decoder_model)]:
                success,producer,detail=_model_producer_status(candidate)
                if not success:
                    raise RuntimeError(
                        f"cached {component} model {candidate.model_id} has a failed producer; "
                        f"producer={getattr(producer,'url',None)}; {detail}"
                    )
            cached["artifact_mode"]=artifact_mode
            cached["encoder_model_id"]=encoder_model.model_id
            cached["decoder_model_id"]=decoder_model.model_id
            print(
                f"Reusing verified S24 artifact ({artifact_mode}): "
                f"encoder={encoder_model.model_id}, decoder={decoder_model.model_id}"
            )
        except Exception as exc:
            print("Cached remote artifact unavailable/invalid; rebuilding:",repr(exc))
            artifact_mode=None
            encoder_model=None
            decoder_model=None
            remote_manifest.pop(label,None)
            atomic_json(remote_manifest,REMOTE_MANIFEST_PATH)

    if encoder_model is None or decoder_model is None:
        meta=export_whisper_components(label,repo)
        graph_names=[f"{slug(label)}_encoder",f"{slug(label)}_decoder"]
        print("Uploading TorchScript source models...")
        src_enc=client.upload_model(meta["encoder_path"],name=f"{slug(label)}_encoder_source")
        src_dec=client.upload_model(meta["decoder_path"],name=f"{slug(label)}_decoder_source")

        common=f"--quantize_full_type float16 --quantize_io --qairt_version {QAIRT_VERSION}"
        compile_options=[
            f"--output_names {','.join(meta['encoder_output_names'])} {common}",
            f"--output_names {','.join(meta['decoder_output_names'])} {common}",
        ]
        if ARTIFACT_BUILD_POLICY=="linked_context":
            print("Submitting compile + link jobs to exact",TARGET_DEVICE_NAME)
            compile_jobs,link_job=client.submit_compile_and_link_jobs(
                models=[src_enc,src_dec],
                device=TARGET_DEVICE,
                name=f"{slug(label)}_s24_whisper",
                input_specs=[meta["encoder_input_spec"],meta["decoder_input_spec"]],
                graph_names=graph_names,
                compile_options=compile_options,
                link_options=f"--qairt_version {QAIRT_VERSION}",
            )
            if link_job is None:
                raise RuntimeError("submit_compile_and_link_jobs returned no LinkJob")
        else:
            print("Submitting two independent QNN DLC compile jobs to exact",TARGET_DEVICE_NAME)
            compile_jobs=_submit_component_compile_jobs(
                client,
                [src_enc,src_dec],
                [meta["encoder_input_spec"],meta["decoder_input_spec"]],
                [meta["encoder_output_names"],meta["decoder_output_names"]],
                TARGET_DEVICE,
                f"{slug(label)}_s24_whisper",
                QAIRT_VERSION,
            )
            link_job=None
        all_build_jobs=[*compile_jobs] + ([link_job] if link_job is not None else [])
        for j in all_build_jobs:
            job_device=getattr(j,"device",None)
            if job_device is not None and getattr(job_device,"name",None)!=TARGET_DEVICE_NAME:
                raise RuntimeError(f"Qualcomm job unexpectedly targeted {job_device}; expected exact {TARGET_DEVICE_NAME}")
        print("Compile jobs:",[j.url for j in compile_jobs])
        if link_job is not None: print("Link job:",link_job.url)

        # IMPORTANT: .wait() returns a JobStatus; it does NOT raise when the remote
        # job fails. Validate every compile and the link explicitly before caching
        # or attempting profile/inference.
        compile_failures=[]
        for idx,j in enumerate(compile_jobs):
            st=j.wait()
            print(f"Compile[{idx}] finished: {_status_line(st)} | {j.url}")
            if not bool(getattr(st,"success",False)):
                compile_failures.append((idx,j,st))
        link_status=None
        if link_job is not None:
            link_status=link_job.wait()
            print(f"Link finished: {_status_line(link_status)} | {link_job.url}")
        if compile_failures:
            bits=[]
            for idx,j,st in compile_failures:
                bits.append(f"compile[{idx}] {j.url}: {_status_line(st)}")
            raise RuntimeError(
                f"Qualcomm compile failed for {label}. " + " | ".join(bits)
            )

        compiled_models=[]
        for idx,j in enumerate(compile_jobs):
            target=j.get_target_model()
            if target is None or not bool(target.wait()):
                raise RuntimeError(f"Successful compile[{idx}] returned no ready QNN DLC target model: {j.url}")
            _require_model_ready(target,f"{label} compile[{idx}]")
            compiled_models.append(target)

        link_attempts=[]
        selected_link_job=None
        linked=None
        if link_job is not None:
            link_attempts.append({
                "job_id":link_job.job_id,
                "job_url":link_job.url,
                "options":f"--qairt_version {QAIRT_VERSION}",
                "status":_status_line(link_status),
                "success":bool(getattr(link_status,"success",False)),
            })
        if link_job is not None and bool(getattr(link_status,"success",False)):
            try:
                linked=link_job.get_target_model()
                if linked is not None and bool(linked.wait()):
                    link_attempts[0]["target_model_id"]=linked.model_id
                    selected_link_job=link_job
                else:
                    link_attempts[0]["success"]=False
                    link_attempts[0]["target_error"]="successful link returned no ready target model"
                    linked=None
            except Exception as exc:
                link_attempts[0]["success"]=False
                link_attempts[0]["target_error"]=repr(exc)
                linked=None

        if link_job is not None and linked is None:
            print("Initial context-binary link failed; retrying with lower HTP optimization levels O=2 then O=1.")
            selected_link_job,linked,retries=_retry_link_jobs(
                client,compiled_models,TARGET_DEVICE,f"{slug(label)}_s24_whisper_link_retry",QAIRT_VERSION
            )
            link_attempts.extend(retries)

        if linked is not None:
            artifact_mode="linked_context"
            encoder_model=linked
            decoder_model=linked
            _require_model_ready(linked,label)
            print("Using linked QNN context binary:",linked.model_id)
        else:
            artifact_mode="separate_qnn_dlc"
            encoder_model,decoder_model=compiled_models
            reason=("all context-binary links failed" if link_job is not None
                    else "direct QNN DLC policy selected; no context link was requested")
            print(
                f"Using the two successful QNN DLC models directly on the S24 NPU ({reason}). "
                f"encoder={encoder_model.model_id}, decoder={decoder_model.model_id}"
            )

        cached={
            "cache_key":key,"repo":repo,"revision":MODEL_REVISIONS[repo],
            "artifact_mode":artifact_mode,
            "source_encoder_model_id":src_enc.model_id,"source_decoder_model_id":src_dec.model_id,
            "compile_job_ids":[j.job_id for j in compile_jobs],
            "compile_job_urls":[j.url for j in compile_jobs],
            "link_job_id":getattr(selected_link_job,"job_id",None),
            "link_job_url":getattr(selected_link_job,"url",None),
            "initial_link_job_id":getattr(link_job,"job_id",None),
            "initial_link_job_url":getattr(link_job,"url",None),
            "link_attempts":link_attempts,
            "linked_model_id":linked.model_id if linked is not None else None,
            "encoder_model_id":encoder_model.model_id,"decoder_model_id":decoder_model.model_id,
            "graph_names":graph_names,
            "encoder_input_spec":meta["encoder_input_spec"],"decoder_input_spec":meta["decoder_input_spec"],
            "encoder_output_names":meta["encoder_output_names"],"decoder_output_names":meta["decoder_output_names"],
            "config":meta["config"],"snapshot":meta["snapshot"],
        }
    else:
        # Rehydrate exact local model contracts from the pinned source revision.
        snap=source_snapshot(repo)
        original=HfWhisper.load_whisper_model(str(snap))
        cfg,enc,dec,enc_spec,dec_spec,enc_out,dec_out=_component_contracts(original)
        cached["snapshot"]=str(snap); cached["config"]=cfg.to_dict()
        cached.setdefault("graph_names",[f"{slug(label)}_encoder",f"{slug(label)}_decoder"])
        cached["encoder_output_names"]=enc_out
        cached["decoder_output_names"]=dec_out
        cached["encoder_input_spec"]=enc_spec
        cached["decoder_input_spec"]=dec_spec
        del enc,dec,original
        gc.collect()
    producers={}
    expected_kind="context" if artifact_mode=="linked_context" else "dlc"
    for component,model in [("encoder",encoder_model),("decoder",decoder_model)]:
        producer=_require_model_ready(model,f"{label} {component}")
        producer_device=getattr(producer,"device",None) if producer is not None else None
        if producer_device is None or getattr(producer_device,"name",None)!=TARGET_DEVICE_NAME:
            raise RuntimeError(
                f"{component.title()} artifact {model.model_id} is not proven to be produced for exact {TARGET_DEVICE_NAME}: "
                f"producer={producer}, producer_device={producer_device}"
            )
        model_type_text=str(getattr(model,"model_type","")).lower()
        if "qnn" not in model_type_text or expected_kind not in model_type_text:
            raise RuntimeError(
                f"Expected QNN {expected_kind} {component} target model; got {getattr(model,'model_type',None)}"
            )
        producers[component]=producer_device.name
        cached[f"{component}_model_type"]=str(getattr(model,"model_type",""))
        cached[f"{component}_model_metadata"]={str(k):str(v) for k,v in (getattr(model,"metadata",{}) or {}).items()}
    cached["producer_device"]=TARGET_DEVICE_NAME
    cached["encoder_producer_device"]=producers["encoder"]
    cached["decoder_producer_device"]=producers["decoder"]
    cached["linked_model_type"]=cached.get("encoder_model_type") if artifact_mode=="linked_context" else None
    contract=inspect_artifact_contract(encoder_model,decoder_model,cached)
    display(contract)
    MODEL_ARTIFACTS[label]={**cached,"encoder_model":encoder_model,"decoder_model":decoder_model}
    remote_manifest[label]={kk:vv for kk,vv in MODEL_ARTIFACTS[label].items() if kk not in {"encoder_model","decoder_model"}}
    atomic_json(remote_manifest,REMOTE_MANIFEST_PATH)

atomic_json({k:{kk:vv for kk,vv in v.items() if kk not in {"encoder_model","decoder_model"}} for k,v in MODEL_ARTIFACTS.items()},REMOTE_MANIFEST_PATH)
print("All optimized model artifacts have SUCCESS producers and are ready on the exact S24 target.")

PROFILE_PATH=ARTIFACT_DIR/"s24_profiles.json"
profile_cache=_safe_json_load(PROFILE_PATH,{})
PROFILE_REQUIRED=True

def _profile_cache_needs_refresh(cached_profile, profile_key: str, required: list[str]) -> bool:
    if not cached_profile or cached_profile.get("profile_key") != profile_key:
        return True
    if cached_profile.get("profile_error"):
        return True
    return any(cached_profile.get(key) is None for key in required)

def _profile_component_complete(profile, component: str) -> bool:
    if not profile or profile.get(f"{component}_latency_us") is None:
        return False
    return any(
        profile.get(f"{component}_{metric}") is not None
        for metric in (
            "inference_peak_memory_bytes",
            "first_load_peak_memory_bytes",
            "warm_load_peak_memory_bytes",
        )
    )

def _qnn_runtime_options(qairt_version: str, graph: str, artifact_mode: str | None) -> str:
    options=f"--compute_unit npu --qairt_version {qairt_version}"
    if artifact_mode!="separate_qnn_dlc":
        options+=f" --qnn_options context_enable_graphs={graph}"
    return options

def _find_profile_metric(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for value in obj.values():
            found=_find_profile_metric(value,key)
            if found is not None:return found
    return None

def _range_upper(value):
    if value is None:return None
    if isinstance(value,(tuple,list)) and len(value)>=2:
        return int(value[1])
    try:return int(value)
    except Exception:return None

def profile_metrics(job):
    status=job.wait()
    if not bool(getattr(status,"success",False)):
        raise RuntimeError(f"Profile job FAILED: {_status_line(status)} | {job.url}")
    report=job.download_profile()
    latency=_find_profile_metric(report,"estimated_inference_time")
    # Support both current Workbench range metrics and legacy scalar names.
    inf_mem=_range_upper(_find_profile_metric(report,"inference_memory_peak_range"))
    first_mem=_range_upper(_find_profile_metric(report,"first_load_memory_peak_range"))
    warm_mem=_range_upper(_find_profile_metric(report,"warm_load_memory_peak_range"))
    if inf_mem is None: inf_mem=_range_upper(_find_profile_metric(report,"estimated_inference_peak_memory"))
    if first_mem is None: first_mem=_range_upper(_find_profile_metric(report,"first_load_peak_memory"))
    if warm_mem is None: warm_mem=_range_upper(_find_profile_metric(report,"warm_load_peak_memory"))
    if latency is None or inf_mem is None or first_mem is None or warm_mem is None:
        for s in client.get_job_summaries(limit=100):
            if getattr(s,"job_id",None)!=job.job_id:
                continue
            if latency is None and getattr(s,"estimated_inference_time",None) is not None:
                latency=int(s.estimated_inference_time)
            if inf_mem is None:
                inf_mem=_range_upper(getattr(s,"inference_memory_peak_range",None))
            if first_mem is None:
                first_mem=_range_upper(getattr(s,"first_load_memory_peak_range",None))
            if warm_mem is None:
                warm_mem=_range_upper(getattr(s,"warm_load_memory_peak_range",None))
            break
    if latency is None:
        raise RuntimeError(f"Cannot find estimated_inference_time for {job.url}")
    if inf_mem is None and first_mem is None and warm_mem is None:
        raise RuntimeError(f"Cannot find peak memory metrics for {job.url}")
    return {
        "latency_us":int(latency),
        "inference_peak_memory_bytes":inf_mem,
        "first_load_peak_memory_bytes":first_mem,
        "warm_load_peak_memory_bytes":warm_mem,
    },report

def _run_profile_job_with_retries(client_obj, model, device, name: str, options: str,
                                  target_device_name: str, max_attempts: int):
    option_candidates=_profile_runtime_option_candidates(options)
    option_index=0
    failures=[]
    for attempt in range(1,max_attempts+1):
        job=None
        attempt_options=option_candidates[option_index]
        try:
            job=client_obj.submit_profile_job(
                model,device=device,name=name,options=attempt_options
            )
            if getattr(getattr(job,"device",None),"name",None)!=target_device_name:
                raise RuntimeError(
                    f"Profile job {getattr(job,'url',None)} is not on exact "
                    f"{target_device_name}: {getattr(job,'device',None)}"
                )
            metrics,report=profile_metrics(job)
            print(
                f"Profile {name} attempt {attempt}/{max_attempts}: SUCCESS | "
                f"{getattr(job,'url',None)}"
            )
            return job,metrics,report
        except Exception as exc:
            detail=(f"attempt {attempt}: options={attempt_options!r} "
                    f"job={getattr(job,'url',None)} error={exc!r}")
            failures.append(detail)
            if _is_memory_allocation_error(exc) and option_index+1<len(option_candidates):
                option_index+=1
            if attempt<max_attempts:
                print(f"WARNING: profile {name} failed; retrying. {detail}")
    raise RuntimeError(
        f"Profile {name} failed after {max_attempts} attempts. " + " | ".join(failures)
    )

def _is_memory_allocation_error(exc) -> bool:
    message=str(exc).upper()
    return "QNN_COMMON_ERROR_MEM_ALLOC" in message or "MEMORY ALLOCATION" in message

def _profile_runtime_option_candidates(options: str) -> list[str]:
    candidates=[options]
    vtcm_option="default_graph_htp_vtcm_size=0"
    if vtcm_option in options:
        return candidates
    if "--qnn_options" in options:
        # AI Hub separates multiple QNN sub-options with semicolons.
        candidates.append(options+f";{vtcm_option}")
    else:
        candidates.append(options+f" --qnn_options {vtcm_option}")
    return candidates

def _compile_single_graph_profile_context(client_obj, source_model, input_spec, output_names,
                                          graph_name: str, device, name: str,
                                          qairt_version: str):
    common=f"--quantize_full_type float16 --quantize_io --qairt_version {qairt_version}"
    compile_jobs,link_job=client_obj.submit_compile_and_link_jobs(
        models=[source_model],
        device=device,
        name=name,
        input_specs=[input_spec],
        graph_names=[graph_name],
        compile_options=[f"--output_names {','.join(output_names)} {common}"],
        link_options=f"--qairt_version {qairt_version}",
    )
    if len(compile_jobs)!=1 or link_job is None:
        raise RuntimeError("Single-graph profile fallback returned an incomplete compile/link job set")
    for job in [compile_jobs[0],link_job]:
        job_device=getattr(job,"device",None)
        if job_device is not None and getattr(job_device,"name",None)!=getattr(device,"name",None):
            raise RuntimeError(f"Profile fallback targeted {job_device}; expected {device}")

    compile_status=compile_jobs[0].wait()
    print(f"Profile fallback compile: {_status_line(compile_status)} | {compile_jobs[0].url}")
    if not bool(getattr(compile_status,"success",False)):
        raise RuntimeError(
            f"Profile fallback compile FAILED: {_status_line(compile_status)} | {compile_jobs[0].url}"
        )
    compiled_model=compile_jobs[0].get_target_model()
    if compiled_model is None or not bool(compiled_model.wait()):
        raise RuntimeError(f"Profile fallback compile returned no ready model: {compile_jobs[0].url}")

    link_status=link_job.wait()
    print(f"Profile fallback link: {_status_line(link_status)} | {link_job.url}")
    link_attempts=[{
        "job_id":link_job.job_id,
        "job_url":link_job.url,
        "options":f"--qairt_version {qairt_version}",
        "status":_status_line(link_status),
        "success":bool(getattr(link_status,"success",False)),
    }]
    target_model=None
    selected_link_job=None
    if bool(getattr(link_status,"success",False)):
        target_model=link_job.get_target_model()
        if target_model is not None and bool(target_model.wait()):
            selected_link_job=link_job
        else:
            target_model=None
            link_attempts[0]["success"]=False
            link_attempts[0]["target_error"]="successful link returned no ready target model"
    if target_model is None:
        selected_link_job,target_model,retries=_retry_link_jobs(
            client_obj,[compiled_model],device,name+"_link_retry",qairt_version
        )
        link_attempts.extend(retries)
    if target_model is None or not bool(target_model.wait()):
        raise RuntimeError(
            "Single-graph QNN context-binary profile fallback failed. "
            + " | ".join(str(row) for row in link_attempts)
        )
    return target_model,{
        "profile_artifact_mode":"single_graph_qnn_context_binary",
        "profile_model_id":target_model.model_id,
        "profile_compile_job_ids":[job.job_id for job in compile_jobs],
        "profile_compile_job_urls":[job.url for job in compile_jobs],
        "profile_link_job_id":getattr(selected_link_job,"job_id",None),
        "profile_link_job_url":getattr(selected_link_job,"url",None),
        "profile_link_attempts":link_attempts,
    }

def _validate_required_profile_rows(rows, required_models):
    rows_by_model={row.get("model"):row for row in rows}
    missing=[]
    for model in required_models:
        row=rows_by_model.get(model)
        if row is None:
            missing.append(f"{model}: profile row")
            continue
        for field in ("encoder_ms","decoder_ms_per_token","peak_ram_mb"):
            value=row.get(field)
            try:
                valid=value is not None and bool(np.isfinite(float(value)))
            except (TypeError,ValueError):
                valid=False
            if not valid:
                missing.append(f"{model}: {field}")
    if missing:
        raise RuntimeError(
            "Required S24 latency/RAM profiling is incomplete: " + "; ".join(missing)
        )

def _validate_required_final_rows(rows, required_models, required_fields):
    rows_by_model={row.get("Model"):row for row in rows}
    missing=[]
    for model in required_models:
        row=rows_by_model.get(model)
        if row is None:
            missing.append(f"{model}: final table row")
            continue
        for field in required_fields:
            value=row.get(field)
            try:
                valid=value is not None and bool(np.isfinite(float(value)))
            except (TypeError,ValueError):
                valid=False
            if not valid:
                missing.append(f"{model}: {field}")
    if missing:
        raise RuntimeError(
            "Final benchmark table has missing required values: " + "; ".join(missing)
        )

PROFILE_ROWS=[]
for label,a in MODEL_ARTIFACTS.items():
    artifact_mode=a["artifact_mode"]
    encoder_model=a["encoder_model"]; decoder_model=a["decoder_model"]
    enc_graph,dec_graph=a["graph_names"]
    _require_model_ready(encoder_model,f"{label} encoder")
    _require_model_ready(decoder_model,f"{label} decoder")
    if artifact_mode=="linked_context":
        pkey=f"{encoder_model.model_id}|{TARGET_DEVICE_NAME}|{DEVICE_FINGERPRINT}|{QAIRT_VERSION}|{QAIRT_FULL_VERSION}|profile-v2"
    else:
        pkey=(f"{artifact_mode}|{encoder_model.model_id}|{decoder_model.model_id}|{TARGET_DEVICE_NAME}|"
              f"{DEVICE_FINGERPRINT}|{QAIRT_VERSION}|{QAIRT_FULL_VERSION}|profile-v3")
    cached_profile=profile_cache.get(label)
    required=["encoder_latency_us","decoder_latency_us"]
    needs_refresh=(
        _profile_cache_needs_refresh(cached_profile,pkey,required)
        or any(
            not _profile_component_complete(cached_profile,component)
            for component in ("encoder","decoder")
        )
    )
    if needs_refresh and not ENABLE_PROFILING:
        raise RuntimeError(
            f"S24 latency/RAM profile is required for {label}, but QAI_ENABLE_PROFILING is disabled."
        )
    if needs_refresh:
        base_vals={
            "profile_key":pkey,
            "artifact_mode":artifact_mode,
            "linked_model_id":a.get("linked_model_id"),
            "encoder_model_id":encoder_model.model_id,
            "decoder_model_id":decoder_model.model_id,
            "profile_status":"requested",
        }
        vals={**base_vals,**cached_profile} if cached_profile and cached_profile.get("profile_key")==pkey else base_vals
        vals["profile_status"]="requested"
        profile_errors=[]
        for component,graph,model in [("encoder",enc_graph,encoder_model),("decoder",dec_graph,decoder_model)]:
            if _profile_component_complete(vals,component):
                print(f"Reusing completed S24 profile for {label}/{component}.")
                continue
            try:
                profile_model=model
                profile_artifact_mode=artifact_mode
                cached_profile_model_id=vals.get(f"{component}_profile_model_id")
                if (vals.get(f"{component}_profile_artifact_mode")=="single_graph_qnn_context_binary"
                        and cached_profile_model_id):
                    try:
                        cached_profile_model=client.get_model(cached_profile_model_id)
                        _require_model_ready(cached_profile_model,f"{label} {component} profile fallback")
                        profile_model=cached_profile_model
                        profile_artifact_mode="single_graph_qnn_context_binary"
                        print(f"Reusing single-graph context profile artifact for {label}/{component}: {cached_profile_model_id}")
                    except Exception as cache_exc:
                        print(f"Cached profile fallback unavailable for {label}/{component}: {cache_exc!r}")

                opts=_qnn_runtime_options(QAIRT_VERSION,graph,profile_artifact_mode)
                try:
                    job,metrics,report=_run_profile_job_with_retries(
                        client,profile_model,TARGET_DEVICE,f"{slug(label)}_{component}_s24",opts,
                        TARGET_DEVICE_NAME,HUB_JOB_RETRIES
                    )
                except Exception as initial_profile_exc:
                    can_fallback=(
                        artifact_mode=="separate_qnn_dlc"
                        and profile_artifact_mode==artifact_mode
                        and _is_memory_allocation_error(initial_profile_exc)
                    )
                    if not can_fallback:
                        raise
                    print(
                        f"QNN DLC profile hit deterministic memory allocation failure for {label}/{component}; "
                        "compiling one graph to an S24-specific context binary for profiling only."
                    )
                    source_model_id=a.get(f"source_{component}_model_id")
                    if not source_model_id:
                        raise RuntimeError(
                            f"Missing source model ID for {label}/{component} profile fallback"
                        ) from initial_profile_exc
                    source_model=client.get_model(source_model_id)
                    if source_model is None or not bool(source_model.wait()):
                        raise RuntimeError(
                            f"Source model {source_model_id} is unavailable for profile fallback"
                        ) from initial_profile_exc
                    profile_model,fallback_evidence=_compile_single_graph_profile_context(
                        client,
                        source_model,
                        a[f"{component}_input_spec"],
                        a[f"{component}_output_names"],
                        graph,
                        TARGET_DEVICE,
                        f"{slug(label)}_{component}_profile_context",
                        QAIRT_VERSION,
                    )
                    for key,value in fallback_evidence.items():
                        vals[f"{component}_{key}"]=value
                    profile_cache[label]=vals
                    atomic_json(profile_cache,PROFILE_PATH)
                    profile_artifact_mode="single_graph_qnn_context_binary"
                    opts=_qnn_runtime_options(QAIRT_VERSION,graph,profile_artifact_mode)
                    job,metrics,report=_run_profile_job_with_retries(
                        client,profile_model,TARGET_DEVICE,
                        f"{slug(label)}_{component}_s24_context_profile",opts,
                        TARGET_DEVICE_NAME,HUB_JOB_RETRIES
                    )
                vals[f"{component}_profile_artifact_mode"]=profile_artifact_mode
                vals[f"{component}_profile_model_id"]=profile_model.model_id
                vals[f"{component}_profile_job_id"]=job.job_id
                vals[f"{component}_profile_url"]=job.url
                vals[f"{component}_latency_us"]=metrics["latency_us"]
                for k,v in metrics.items():
                    if k!="latency_us": vals[f"{component}_{k}"]=v
            except Exception as exc:
                profile_errors.append(f"{component}: {exc!r}")
                print(f"ERROR: required S24 profile failed for {label}/{component}. {exc!r}")
                vals[f"{component}_latency_us"]=None
                vals[f"{component}_profile_job_id"]=None
                vals[f"{component}_profile_url"]=None
                for k in ["inference_peak_memory_bytes","first_load_peak_memory_bytes","warm_load_peak_memory_bytes"]:
                    vals[f"{component}_{k}"]=None
                vals["profile_status"]="failed"
                vals["profile_error"]=" | ".join(profile_errors)
                profile_cache[label]=vals
                atomic_json(profile_cache,PROFILE_PATH)
                if PROFILE_REQUIRED: raise
        vals["profile_status"]="success" if not profile_errors else "failed"
        vals["profile_error"]=" | ".join(profile_errors) if profile_errors else None
        profile_cache[label]=vals
        atomic_json(profile_cache,PROFILE_PATH)
    v=profile_cache[label]
    enc_ms=(v.get("encoder_latency_us")/1000) if v.get("encoder_latency_us") is not None else np.nan
    dec_ms=(v.get("decoder_latency_us")/1000) if v.get("decoder_latency_us") is not None else np.nan
    mem_values=[v.get(f"{c}_{m}") for c in ["encoder","decoder"] for m in [
        "inference_peak_memory_bytes","first_load_peak_memory_bytes","warm_load_peak_memory_bytes"]]
    mem_values=[x for x in mem_values if x is not None]
    peak_ram_mb=(max(mem_values)/(1024**2)) if mem_values else np.nan
    PROFILE_ROWS.append({
        "model":label,"artifact_mode":artifact_mode,"linked_model_id":a.get("linked_model_id"),
        "encoder_model_id":encoder_model.model_id,"decoder_model_id":decoder_model.model_id,
        "device":TARGET_DEVICE_NAME,
        "encoder_ms":enc_ms,"decoder_ms_per_token":dec_ms,
        "encoder_plus_first_decoder_ms":enc_ms+dec_ms if np.isfinite(enc_ms) and np.isfinite(dec_ms) else np.nan,
        "max_200_step_compute_ms":enc_ms+(MEAN_DECODE_LEN-1)*dec_ms if np.isfinite(enc_ms) and np.isfinite(dec_ms) else np.nan,
        "peak_ram_mb":peak_ram_mb,
        "encoder_inference_peak_memory_mb":(v.get("encoder_inference_peak_memory_bytes")/(1024**2)) if v.get("encoder_inference_peak_memory_bytes") is not None else np.nan,
        "decoder_inference_peak_memory_mb":(v.get("decoder_inference_peak_memory_bytes")/(1024**2)) if v.get("decoder_inference_peak_memory_bytes") is not None else np.nan,
        "encoder_profile_artifact_mode":v.get("encoder_profile_artifact_mode"),
        "decoder_profile_artifact_mode":v.get("decoder_profile_artifact_mode"),
        "encoder_profile_model_id":v.get("encoder_profile_model_id"),
        "decoder_profile_model_id":v.get("decoder_profile_model_id"),
        "encoder_profile_job_id":v.get("encoder_profile_job_id"),"decoder_profile_job_id":v.get("decoder_profile_job_id"),
        "profile_error":v.get("profile_error"),
    })
PROFILE_DF=pd.DataFrame(PROFILE_ROWS)
_validate_required_profile_rows(PROFILE_ROWS,list(MODEL_IDS))
PROFILE_MAP=PROFILE_DF.set_index("model").to_dict("index")
PROFILE_DF.to_csv(RESULT_DIR/"s24_model_speed_memory_profile.csv",index=False)
display(PROFILE_DF)

def np_dtype_from_hub(dtype: Any):
    s=str(dtype).lower()
    for key,d in [("float16",np.float16),("float32",np.float32),("int32",np.int32),("int64",np.int64),
                  ("uint8",np.uint8),("int8",np.int8),("uint16",np.uint16),("int16",np.int16)]:
        if key in s:return d
    raise TypeError(f"Unsupported Workbench tensor dtype: {dtype}")

GRAPH_CONTRACTS={}
for _label,_a in MODEL_ARTIFACTS.items():
    _enc_graph,_dec_graph=_a["graph_names"]
    GRAPH_CONTRACTS[_enc_graph]={
        "inputs":_normalize_compile_input_spec(_a["encoder_input_spec"]),
        "outputs":list(_a["encoder_output_names"]),
    }
    GRAPH_CONTRACTS[_dec_graph]={
        "inputs":_normalize_compile_input_spec(_a["decoder_input_spec"]),
        "outputs":list(_a["decoder_output_names"]),
    }

def graph_input_specs(model,graph):
    remote=getattr(model,"input_spec",{}) or {}
    specs=remote.get(graph)
    if not specs and len(remote)==1 and None in remote:
        specs=remote[None]
    if specs:
        return list(specs)
    if graph not in GRAPH_CONTRACTS:
        raise KeyError(f"No local compile contract for graph {graph!r}")
    return GRAPH_CONTRACTS[graph]["inputs"]

def graph_output_names(model,graph):
    remote=getattr(model,"output_spec",{}) or {}
    specs=remote.get(graph)
    if not specs and len(remote)==1 and None in remote:
        specs=remote[None]
    if specs:
        return [t.name for t in specs]
    if graph not in GRAPH_CONTRACTS:
        raise KeyError(f"No local compile contract for graph {graph!r}")
    return GRAPH_CONTRACTS[graph]["outputs"]

def cast_for_tensor(arr: np.ndarray, ts):
    x=np.asarray(arr,dtype=np_dtype_from_hub(ts.dtype))
    expected=tuple(int(v) for v in ts.shape)
    if tuple(x.shape)!=expected:
        raise ValueError(f"{ts.name}: got shape {x.shape}, expected {expected}")
    return np.ascontiguousarray(x)

def infer_graph(model, graph: str, examples: list[dict[str,np.ndarray]], job_name: str,
                artifact_mode: str | None=None, component: str | None=None,
                linked_model_id: str | None=None):
    if not examples: return [],0.0,None
    ins=graph_input_specs(model,graph)
    expected_names=[t.name for t in ins]
    for ex in examples:
        if set(ex)!=set(expected_names):
            raise KeyError(f"{graph} input mismatch: expected {expected_names}; got {list(ex)}")
    payload={t.name:[cast_for_tensor(ex[t.name],t) for ex in examples] for t in ins}
    opts=_qnn_runtime_options(QAIRT_VERSION,graph,artifact_mode)
    t0=time.perf_counter()
    job,raw=_run_inference_job_with_retries(
        client,model,TARGET_DEVICE,payload,job_name,opts,TARGET_DEVICE_NAME,HUB_JOB_RETRIES
    )
    wall=time.perf_counter()-t0
    out_names=graph_output_names(model,graph)
    missing=[name for name in out_names if name not in raw]
    if missing:
        raise RuntimeError(f"Workbench output-name contract changed for {graph}; missing {missing}; returned {list(raw)}")
    k=len(examples)
    per_example=[]
    for i in range(k):
        row={}
        for name in out_names:
            vals=raw[name]
            if len(vals)!=k: raise RuntimeError(f"Output {name} returned {len(vals)} entries for {k} inputs")
            row[name]=np.asarray(vals[i])
        per_example.append(row)
    append_job_evidence({
        "kind": "inference",
        "status": "success",
        "job_id": job.job_id,
        "job_url": job.url,
        "job_name": job_name,
        "device": getattr(getattr(job, "device", None), "name", None),
        "artifact_mode": artifact_mode,
        "component": component,
        "artifact_model_id": getattr(model, "model_id", None),
        "linked_model_id": linked_model_id,
        "graph": graph,
        "batch_size": len(examples),
        "compute_unit_requested": "npu",
        "qairt_api_version": QAIRT_VERSION,
        "wall_seconds": wall,
    })
    return per_example,wall,job.job_id

@dataclass
class DecodeState:
    tokens: list[int]
    self_cache: dict[str,np.ndarray]
    cross_cache: dict[str,np.ndarray]
    attention_mask: np.ndarray
    position_ids: np.ndarray
    done: bool=False
    truncated: bool=False

class S24WhisperRuntime:
    def __init__(self,label: str):
        self.label=label; self.a=MODEL_ARTIFACTS[label]
        self.encoder_model=self.a["encoder_model"]; self.decoder_model=self.a["decoder_model"]
        self.enc_graph,self.dec_graph=self.a["graph_names"]
        self.snapshot=self.a["snapshot"]
        self.processor=AutoProcessor.from_pretrained(self.snapshot)
        try:self.gen=GenerationConfig.from_pretrained(self.snapshot)
        except Exception:self.gen=GenerationConfig()
        self.config=WhisperConfig.from_dict(self.a["config"])
        self.enc_out_names=self.a["encoder_output_names"]
        self.dec_out_names=self.a["decoder_output_names"]
        if set(graph_output_names(self.encoder_model,self.enc_graph)) != set(self.enc_out_names):
            raise RuntimeError("Encoder target output names differ from Qualcomm source contract")
        if set(graph_output_names(self.decoder_model,self.dec_graph)) != set(self.dec_out_names):
            raise RuntimeError("Decoder target output names differ from Qualcomm source contract")
        self.dec_input_spec={t.name:t for t in graph_input_specs(self.decoder_model,self.dec_graph)}
        expected_dec={"logits"} | {f"{p}_cache_self_{i}_out" for i in range(self.config.decoder_layers) for p in ("k","v")}
        if set(self.dec_out_names)!=expected_dec:
            raise RuntimeError(f"Unexpected Qualcomm Whisper decoder outputs for {label}: {self.dec_out_names}")
        expected_cross={f"{p}_cache_cross_{i}" for i in range(self.config.decoder_layers) for p in ("k","v")}
        if set(self.enc_out_names)!=expected_cross:
            raise RuntimeError(f"Unexpected Qualcomm Whisper encoder outputs for {label}: {self.enc_out_names}")

    def forced_prompt(self,language: str):
        fn=getattr(self.processor,"get_decoder_prompt_ids",None) or getattr(self.processor.tokenizer,"get_decoder_prompt_ids",None)
        if fn is None: raise RuntimeError("Installed Transformers tokenizer has no get_decoder_prompt_ids")
        pairs=fn(language=language,task="transcribe",no_timestamps=True)
        return {int(pos):int(tok) for pos,tok in pairs}

    def _new_state(self,cross_values):
        if not isinstance(cross_values,dict):
            raise TypeError(f"Encoder output must be a name->tensor dict, got {type(cross_values)}")
        cross={name:cross_values[name] for name in self.enc_out_names}
        self_cache={}
        for name,ts in self.dec_input_spec.items():
            if re.fullmatch(r"[kv]_cache_self_\d+_in",name):
                self_cache[name]=np.zeros(tuple(ts.shape),dtype=np_dtype_from_hub(ts.dtype))
        mask_ts=self.dec_input_spec["attention_mask"]
        mask=np.full(tuple(mask_ts.shape),float(getattr(self.config,"mask_neg",-100.0)),dtype=np_dtype_from_hub(mask_ts.dtype))
        pos_ts=self.dec_input_spec["position_ids"]
        pos=np.zeros(tuple(pos_ts.shape),dtype=np_dtype_from_hub(pos_ts.dtype))
        sot=int(getattr(self.config,"decoder_start_token_id",None) or self.processor.tokenizer.bos_token_id)
        return DecodeState(tokens=[sot],self_cache=self_cache,cross_cache=cross,attention_mask=mask,position_ids=pos)

    def _choose_next(self,logits,next_position:int,forced:dict[int,int],first_free_position:int):
        scores=np.asarray(logits).reshape(-1).astype(np.float32,copy=True)
        if next_position in forced:return forced[next_position]
        suppress=list(getattr(self.gen,"suppress_tokens",[]) or [])
        begin=list(getattr(self.gen,"begin_suppress_tokens",[]) or []) if next_position==first_free_position else []
        for tok in suppress+begin:
            tok=int(tok)
            if 0<=tok<scores.size:scores[tok]=-np.inf
        return int(np.argmax(scores))

    def transcribe_batch(self,waves:list[np.ndarray],language:str,batch_tag:str):
        features=[]
        for wave in waves:
            wave=np.asarray(wave,np.float32)
            f=self.processor.feature_extractor(wave,sampling_rate=SR,return_tensors="np")["input_features"]
            if tuple(f.shape)!=(1,self.config.num_mel_bins,3000):
                raise RuntimeError(f"Unexpected Whisper features {f.shape}")
            features.append({"input_features":f})
        enc_values,enc_wall,enc_job=infer_graph(
            self.encoder_model,self.enc_graph,features,f"{slug(self.label)}_{batch_tag}_enc",
            artifact_mode=self.a["artifact_mode"],component="encoder",linked_model_id=self.a.get("linked_model_id")
        )
        states=[self._new_state(v) for v in enc_values]
        forced=self.forced_prompt(language)
        first_free=max(forced.keys(),default=0)+1
        eos=int(getattr(self.config,"eos_token_id",None) or self.processor.tokenizer.eos_token_id)
        decoder_wall=0.0; decoder_jobs=[]
        per_state_last_decoder_job=[None]*len(states)
        per_state_decoder_job_count=[0]*len(states)

        for n in range(MEAN_DECODE_LEN-1):
            active=[i for i,s in enumerate(states) if not s.done]
            if not active:break
            inputs=[]
            for i in active:
                s=states[i]
                s.attention_mask[...,MEAN_DECODE_LEN-n-1]=0.0
                ex={"input_ids":np.array([[s.tokens[-1]]],dtype=np.int32),"attention_mask":s.attention_mask}
                ex.update(s.self_cache); ex.update(s.cross_cache); ex["position_ids"]=s.position_ids
                inputs.append(ex)
            outs,wall,jid=infer_graph(
                self.decoder_model,self.dec_graph,inputs,f"{slug(self.label)}_{batch_tag}_dec_{n:03d}",
                artifact_mode=self.a["artifact_mode"],component="decoder",linked_model_id=self.a.get("linked_model_id")
            )
            decoder_wall+=wall; decoder_jobs.append(jid)
            for state_idx,vals in zip(active,outs):
                s=states[state_idx]
                per_state_last_decoder_job[state_idx]=jid
                per_state_decoder_job_count[state_idx]+=1
                if "logits" not in vals:
                    raise RuntimeError(f"Decoder output missing logits; got {list(vals)}")
                token=self._choose_next(vals["logits"],n+1,forced,first_free)
                s.tokens.append(token)
                for block in range(self.config.decoder_layers):
                    k_out=f"k_cache_self_{block}_out"
                    v_out=f"v_cache_self_{block}_out"
                    if k_out not in vals or v_out not in vals:
                        raise RuntimeError(f"Decoder output missing cache tensors for block {block}; got {list(vals)}")
                    s.self_cache[f"k_cache_self_{block}_in"]=vals[k_out]
                    s.self_cache[f"v_cache_self_{block}_in"]=vals[v_out]
                s.position_ids=s.position_ids+1
                if token==eos:s.done=True
        for s in states:
            if not s.done:s.truncated=True
        texts=[self.processor.tokenizer.decode(s.tokens,skip_special_tokens=True).strip() for s in states]
        total_wall=enc_wall+decoder_wall
        return [
            {"raw_prediction":text,"decode_steps":len(s.tokens)-1,"truncated":s.truncated,
             "encoder_job_id":enc_job,
             "last_decoder_job_id":per_state_last_decoder_job[i],
             "decoder_job_count":per_state_decoder_job_count[i],
             "api_batch_wall_s":total_wall,
             "api_wall_s_share":total_wall/max(1,len(states))}
            for i,(text,s) in enumerate(zip(texts,states))
        ]

RUNTIMES={label:S24WhisperRuntime(label) for label in MODEL_IDS}
print("Runtime contracts initialized for:",list(RUNTIMES))


def _mono_audio_array(data):
    x=np.asarray(data,np.float32).squeeze()
    if x.ndim==2:
        if x.shape[0]<=8:
            x=x.mean(axis=0,dtype=np.float32)
        elif x.shape[1]<=8:
            x=x.mean(axis=1,dtype=np.float32)
    return np.ascontiguousarray(x,dtype=np.float32)

def _resample_audio(data, source_rate: int):
    x=_mono_audio_array(data)
    source_rate=int(source_rate)
    if source_rate==SR:
        return x,SR
    divisor=math.gcd(source_rate,SR)
    x=resample_poly(x,SR//divisor,source_rate//divisor)
    return np.ascontiguousarray(x,dtype=np.float32),SR

def audio_array(audio_obj):
    # Datasets <=3 dictionary Audio representation (the original notebooks' representation).
    if isinstance(audio_obj,dict) and "array" in audio_obj:
        return _resample_audio(audio_obj["array"],audio_obj["sampling_rate"])
    # Audio(decode=False) avoids the datasets 4.x torchcodec dependency on Windows.
    if isinstance(audio_obj,dict) and ("bytes" in audio_obj or "path" in audio_obj):
        audio_bytes=audio_obj.get("bytes")
        source=io.BytesIO(bytes(audio_bytes)) if audio_bytes is not None else audio_obj.get("path")
        if source is None:
            raise ValueError("Audio record has neither bytes nor path")
        data,sr=sf.read(source,dtype="float32",always_2d=True)
        return _resample_audio(data,sr)
    # Datasets 4.x torchcodec-backed AudioDecoder compatibility.
    if hasattr(audio_obj,"get_all_samples"):
        samples=audio_obj.get_all_samples()
        data=getattr(samples,"data",None); sr=getattr(samples,"sample_rate",None)
        if hasattr(data,"numpy"):data=data.numpy()
        return _resample_audio(data,sr)
    raise TypeError(f"Unsupported datasets Audio object: {type(audio_obj)}")

def validate_wave(x,sr):
    x=np.asarray(x,np.float32)
    if x.ndim!=1 or int(sr)!=SR or len(x)==0 or not np.isfinite(x).all(): raise ValueError("invalid mono/16k/finite waveform")
    dur=len(x)/SR
    if dur>MAX_AUDIO_SECONDS: raise ValueError(f"duration {dur:.3f}s exceeds {MAX_AUDIO_SECONDS}s")
    return np.ascontiguousarray(x),dur

def checkpoint_signature(benchmark_id,label,dataset_revision,language):
    a=MODEL_ARTIFACTS[label]
    if a["artifact_mode"]=="linked_context":
        payload={"benchmark":benchmark_id,"model":label,"model_revision":a["revision"],"linked_model_id":a["encoder_model"].model_id,
                 "dataset_revision":dataset_revision,"language":language,"decode_len":MEAN_DECODE_LEN,
                 "target_device":TARGET_DEVICE_NAME,"device_fingerprint":DEVICE_FINGERPRINT,
                 "qairt_api_version":QAIRT_VERSION,"qairt_full_version":QAIRT_FULL_VERSION,
                 "prediction_contract_version":"2026-08-unified-v2"}
    else:
        payload={"benchmark":benchmark_id,"model":label,"model_revision":a["revision"],
                 "artifact_mode":a["artifact_mode"],"encoder_model_id":a["encoder_model"].model_id,
                 "decoder_model_id":a["decoder_model"].model_id,
                 "dataset_revision":dataset_revision,"language":language,"decode_len":MEAN_DECODE_LEN,
                 "target_device":TARGET_DEVICE_NAME,"device_fingerprint":DEVICE_FINGERPRINT,
                 "qairt_api_version":QAIRT_VERSION,"qairt_full_version":QAIRT_FULL_VERSION,
                 "prediction_contract_version":"2026-08-component-artifacts-v3"}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

def run_predictions(benchmark_id:str, metadata:pd.DataFrame, wave_getter, language:str, dataset_revision:str):
    all_frames=[]
    for label in MODEL_IDS:
        cp=CHECKPOINT_DIR/f"{benchmark_id}__{slug(label)}.csv"
        side=cp.with_suffix(".json")
        sig,payload=checkpoint_signature(benchmark_id,label,dataset_revision,language)
        done=pd.DataFrame()
        if cp.exists() and side.exists():
            info=json.loads(side.read_text())
            if info.get("signature")!=sig:
                raise RuntimeError(f"Stale checkpoint contract: {cp}. Rename/delete it or restore the matching artifact/config.")
            done=pd.read_csv(cp,keep_default_na=False)
            if done.sample_key.duplicated().any():raise RuntimeError(f"Duplicate checkpoint keys: {cp}")
        done_keys=set(done.sample_key.astype(str)) if len(done) else set()
        pending=metadata[~metadata.sample_key.astype(str).isin(done_keys)].copy()
        print(f"{benchmark_id} | {label}: {len(done)}/{len(metadata)} cached; {len(pending)} pending")
        rows=done.to_dict("records") if len(done) else []
        def _transcribe_adaptive(batch, start_tag):
            waves=[wave_getter(r) for _,r in batch.iterrows()]
            try:
                return RUNTIMES[label].transcribe_batch(waves,language,batch_tag=start_tag)
            except Exception as exc:
                if len(batch)<=1:
                    raise
                mid=len(batch)//2
                print(f"Microbatch {len(batch)} failed for {benchmark_id}/{label}; retrying {mid}+{len(batch)-mid}. Reason: {exc!r}")
                left=_transcribe_adaptive(batch.iloc[:mid],start_tag+"_a")
                right=_transcribe_adaptive(batch.iloc[mid:],start_tag+"_b")
                return left+right

        for start in tqdm(range(0,len(pending),HUB_MICROBATCH),desc=f"{benchmark_id}/{label}"):
            batch=pending.iloc[start:start+HUB_MICROBATCH]
            preds=_transcribe_adaptive(batch,f"{slug(benchmark_id)}_{start:06d}")
            prof=PROFILE_MAP[label]
            for (_,r),pred in zip(batch.iterrows(),preds):
                est_ms=(prof["encoder_ms"]+pred["decode_steps"]*prof["decoder_ms_per_token"]
                        if np.isfinite(prof.get("encoder_ms",np.nan)) and np.isfinite(prof.get("decoder_ms_per_token",np.nan))
                        else np.nan)
                artifact=MODEL_ARTIFACTS[label]
                rows.append({"sample_key":r.sample_key,"model":label,**pred,"estimated_device_compute_ms":est_ms,
                             "artifact_mode":artifact["artifact_mode"],
                             "encoder_model_id":artifact["encoder_model"].model_id,
                             "decoder_model_id":artifact["decoder_model"].model_id,
                             "linked_model_id":artifact.get("linked_model_id")})
            frame=pd.DataFrame(rows)
            # stable output order follows metadata, regardless of prior smoke/full order.
            order={k:i for i,k in enumerate(metadata.sample_key.astype(str))}
            frame["__order"]=frame.sample_key.astype(str).map(order)
            frame=frame.sort_values("__order").drop(columns="__order").reset_index(drop=True)
            atomic_csv(frame,cp); atomic_json({"signature":sig,**payload},side)
            rows=frame.to_dict("records")
        final=pd.DataFrame(rows)
        wanted=set(metadata.sample_key.astype(str)); final=final[final.sample_key.astype(str).isin(wanted)].copy()
        if set(final.sample_key.astype(str))!=wanted:raise RuntimeError(f"Incomplete checkpoint after run: {benchmark_id}/{label}")
        all_frames.append(final)
    return pd.concat(all_frames,ignore_index=True)

def detailed_metrics(metadata,predictions,normalizer):
    d=metadata.merge(predictions,on="sample_key",how="inner",validate="one_to_many")
    d["normalized_prediction"]=d.raw_prediction.map(normalizer)
    wc=[];cc=[]
    for r in d.itertuples():
        wc.append(word_counts(r.normalized_reference,r.normalized_prediction));cc.append(char_counts(r.normalized_reference,r.normalized_prediction))
    return pd.concat([d.reset_index(drop=True),pd.DataFrame(wc),pd.DataFrame(cc)],axis=1)


_FLEURS_CACHE={}
def load_fleurs(locale:str,language_name:str,normalizer):
    key=locale
    if key in _FLEURS_CACHE:return _FLEURS_CACHE[key]
    rev=DATASET_REVISIONS["google/fleurs"]
    ds=load_dataset("google/fleurs",locale,split="test",revision=rev)
    ds=ds.cast_column("audio",Audio(decode=False))
    valid=[];excluded=[]
    limit=SMOKE_N if RUN_MODE=="smoke" else (BENCHMARK_N if RUN_MODE=="benchmark" else None)
    for idx,row in enumerate(tqdm(ds,desc=f"Validate FLEURS {locale}")):
        sid=str(row.get("id",idx));ref=str(row.get("raw_transcription") or row.get("transcription") or "")
        try:
            x,sr=audio_array(row["audio"]);x,dur=validate_wave(x,sr);nr=normalizer(ref)
            if not nr:raise ValueError("empty normalized reference")
            valid.append({"dataset_index":idx,"sample_key":f"fleurs::{locale}::{idx}::{sid}","sample_id":sid,
                          "duration_sec":dur,"raw_reference":ref,"normalized_reference":nr})
            if limit and len(valid)>=limit:break
        except Exception as exc:excluded.append({"dataset_index":idx,"sample_id":sid,"reason":repr(exc)})
    meta=pd.DataFrame(valid)
    if meta.empty:raise RuntimeError(f"No valid FLEURS {locale} samples")
    def wave(r):
        x,sr=audio_array(ds[int(r.dataset_index)]["audio"]);x,_=validate_wave(x,sr);return x
    _FLEURS_CACHE[key]=(ds,meta,wave,pd.DataFrame(excluded))
    print(language_name,{"evaluated":len(meta),"audio_hours":meta.duration_sec.sum()/3600,"excluded_seen":len(excluded),"revision":rev})
    return _FLEURS_CACHE[key]

fleurs_vi_ds,FLEURS_VI_META,fleurs_vi_wave,FLEURS_VI_EXCLUDED=load_fleurs("vi_vn","Vietnamese",normalize_vi)
fleurs_en_ds,FLEURS_EN_META,fleurs_en_wave,FLEURS_EN_EXCLUDED=load_fleurs("en_us","English",normalize_en)


ALL_SUMMARIES=[]; ALL_DETAILS={}
if RUN["fleurs_vi_clean"]:
    pred=run_predictions("fleurs_vi_clean",FLEURS_VI_META,fleurs_vi_wave,"vi",DATASET_REVISIONS["google/fleurs"])
    det=detailed_metrics(FLEURS_VI_META,pred,normalize_vi)
    fairness=pairwise_fairness(det,FLEURS_VI_META)
    audit=normalization_audit(FLEURS_VI_META,det)
    top=top_error_audit(det,10)
    summ=corpus_summary(det,["model"]);summ.insert(0,"benchmark","FLEURS Vietnamese clean");summ["split"]="test";summ["condition"]="clean"
    det.to_csv(RESULT_DIR/"fleurs_vi_clean_detailed.csv",index=False)
    summ.to_csv(RESULT_DIR/"fleurs_vi_clean_summary.csv",index=False)
    fairness.to_csv(RESULT_DIR/"fleurs_vi_clean_pairwise_fairness.csv",index=False)
    audit.to_csv(RESULT_DIR/"fleurs_vi_clean_normalization_audit.csv",index=False)
    top.to_csv(RESULT_DIR/"fleurs_vi_clean_top_errors.csv",index=False)
    FLEURS_VI_EXCLUDED.to_csv(RESULT_DIR/"fleurs_vi_clean_excluded_samples.csv",index=False)
    ALL_DETAILS["fleurs_vi_clean"]=det;ALL_SUMMARIES.append(summ)
    display(summ[["model","wer_percent","cer_percent","S","D","I","empty_predictions","estimated_device_rtf","estimated_device_x_realtime"]])
    display(top.head(min(30,len(top))))
    print_wer_ranking(summ,"Vietnamese clean accuracy ranking")


DEMAND_ARCHIVES=["DWASHING_16k.zip","DKITCHEN_16k.zip","TMETRO_16k.zip","TBUS_16k.zip","PSTATION_16k.zip","OOFFICE_16k.zip"]
DEMAND_DIR=DATA_DIR/"demand";DEMAND_DIR.mkdir(parents=True,exist_ok=True)
NOISE_PREP_DIR=DATA_DIR/"fleurs_vi_demand_0db";NOISE_PREP_DIR.mkdir(parents=True,exist_ok=True)

def safe_extract(zf,dst):
    root=dst.resolve()
    for m in zf.infolist():
        target=(dst/m.filename).resolve()
        if root not in target.parents and target!=root:raise RuntimeError("Unsafe ZIP member")
    zf.extractall(dst)

def ensure_demand():
    record=json.load(urlopen("https://zenodo.org/api/records/1227121"));remote={f["key"]:f for f in record["files"]}
    for name in DEMAND_ARCHIVES:
        if name not in remote:raise RuntimeError(f"Official DEMAND archive missing: {name}")
        path=DEMAND_DIR/name;expected=remote[name]["checksum"].split(":",1)[1]
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest()!=expected:
            print("Downloading",name);path.write_bytes(urlopen(remote[name]["links"]["self"]).read())
        if hashlib.md5(path.read_bytes()).hexdigest()!=expected:raise RuntimeError(f"MD5 failed: {name}")
        out=DEMAND_DIR/name[:-4]
        if not out.exists():
            out.mkdir();
            with zipfile.ZipFile(path) as z:safe_extract(z,out)
    files=sorted(p for p in DEMAND_DIR.rglob("*.wav") if p.is_file())
    if not files:raise RuntimeError("No DEMAND WAV files")
    return files

def stable_rng(key):
    seed=int.from_bytes(hashlib.sha256(f"{SEED}|{key}".encode()).digest()[:8],"little")
    return np.random.default_rng(seed)
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def read_noise_segment(path,n,rng):
    z,sr=sf.read(path,dtype="float32",always_2d=True);ch=int(rng.integers(z.shape[1]));z=z[:,ch]
    if sr!=SR:z=resample_poly(z,SR,sr).astype(np.float32)
    if len(z)==0 or not np.isfinite(z).all():raise ValueError("bad noise")
    if len(z)<n:z=np.tile(z,int(np.ceil(n/len(z))))
    off=int(rng.integers(0,len(z)-n+1));return np.ascontiguousarray(z[off:off+n]),ch,off

NOISE_META=None
if RUN["fleurs_vi_noise_0db"]:
    noise_files=ensure_demand();rows=[]
    for r in tqdm(FLEURS_VI_META.itertuples(index=False),total=len(FLEURS_VI_META),desc="Prepare exact 0 dB pairs"):
        key=f"{int(r.dataset_index):06d}";cp=NOISE_PREP_DIR/f"{key}_clean.wav";npth=NOISE_PREP_DIR/f"{key}_0db.wav"
        info_path=NOISE_PREP_DIR/f"{key}.json"
        rebuild=True
        if cp.exists() and npth.exists() and info_path.exists():
            info=json.loads(info_path.read_text())
            rebuild=not (info.get("dataset_revision")==DATASET_REVISIONS["google/fleurs"] and info.get("target_snr_db")==TARGET_SNR_DB)
        if rebuild:
            clean0=fleurs_vi_wave(pd.Series(r._asdict()));rng=stable_rng(r.dataset_index);path=noise_files[int(rng.integers(len(noise_files)))]
            noise,ch,off=read_noise_segment(path,len(clean0),rng);rx,rn=rms(clean0),rms(noise)
            if rx<1e-7 or rn<1e-7:raise RuntimeError("near-zero clean/noise RMS")
            alpha=rx/(rn*10**(TARGET_SNR_DB/20));mix=clean0+alpha*noise
            peak=max(float(np.max(np.abs(clean0))),float(np.max(np.abs(mix))))
            gain=min(1.0,0.99/max(peak,1e-12));sf.write(cp,clean0*gain,SR,subtype="PCM_16");sf.write(npth,mix*gain,SR,subtype="PCM_16")
            cq,_=sf.read(cp,dtype="float32");yq,_=sf.read(npth,dtype="float32");measured=20*np.log10(rms(cq)/rms(yq-cq))
            if abs(measured-TARGET_SNR_DB)>.1:raise AssertionError(f"SNR tolerance failed: {measured}")
            info={"dataset_revision":DATASET_REVISIONS["google/fleurs"],"target_snr_db":TARGET_SNR_DB,"measured_snr_db":measured,
                  "noise_file":str(path),"noise_channel":ch,"noise_offset_samples":off,"alpha":alpha,"shared_gain":gain}
            atomic_json(info,info_path)
        else:measured=float(info["measured_snr_db"])
        rows.append({"sample_key":r.sample_key,"dataset_index":r.dataset_index,"clean_path":str(cp),"noisy_path":str(npth),
                     "measured_snr_db":measured,"noisy_sha256":hashlib.sha256(npth.read_bytes()).hexdigest()})
    NOISE_META=pd.DataFrame(rows)
    assert NOISE_META.measured_snr_db.sub(TARGET_SNR_DB).abs().max()<=.1
    NOISE_META.to_csv(RESULT_DIR/"fleurs_vi_noise_metadata.csv",index=False)
    print("Prepared",len(NOISE_META),"shared clean/noisy pairs; max |SNR error| =",NOISE_META.measured_snr_db.abs().max())


if RUN["fleurs_vi_noise_0db"]:
    base=FLEURS_VI_META.merge(NOISE_META,on=["sample_key","dataset_index"],validate="one_to_one")
    def clean_pair_wave(r):x,sr=sf.read(r.clean_path,dtype="float32");return validate_wave(x,sr)[0]
    def noisy_pair_wave(r):x,sr=sf.read(r.noisy_path,dtype="float32");return validate_wave(x,sr)[0]
    summaries=[];detail_parts=[]
    for cond,getter,bid in [("Clean paired",clean_pair_wave,"fleurs_vi_noise_clean"),("0 dB",noisy_pair_wave,"fleurs_vi_noise_0db")]:
        pred=run_predictions(bid,base,getter,"vi",DATASET_REVISIONS["google/fleurs"])
        det=detailed_metrics(base,pred,normalize_vi);det["condition"]=cond;detail_parts.append(det)
        s=corpus_summary(det,["model"]);s["condition"]=cond;summaries.append(s)
    noise_det=pd.concat(detail_parts,ignore_index=True)
    ns=pd.concat(summaries,ignore_index=True)
    # Prove all three models saw exactly the same sample set in each condition.
    for cond,g in noise_det.groupby("condition",sort=False):
        pairwise_fairness(g,base).to_csv(RESULT_DIR/f"fleurs_vi_noise_{slug(cond)}_pairwise_fairness.csv",index=False)
    pivot=ns.pivot(index="model",columns="condition",values="wer_percent").reset_index()
    pivot["wer_degradation_pp"]=pivot["0 dB"]-pivot["Clean paired"]
    for metric,label in [("S","S"),("D","D"),("I","I"),("empty_predictions","Empty")]:
        ep=ns.pivot(index="model",columns="condition",values=metric)
        pivot[f"Clean {label}"]=pivot["model"].map(ep["Clean paired"])
        pivot[f"0 dB {label}"]=pivot["model"].map(ep["0 dB"])
    noise_det.to_csv(RESULT_DIR/"fleurs_vi_noise_detailed.csv",index=False)
    pivot.to_csv(RESULT_DIR/"fleurs_vi_noise_robustness_summary.csv",index=False)
    NOISE_META.to_csv(RESULT_DIR/"fleurs_vi_noise_metadata.csv",index=False)
    FLEURS_VI_EXCLUDED.to_csv(RESULT_DIR/"fleurs_vi_noise_excluded_samples.csv",index=False)
    top_error_audit(noise_det,10).to_csv(RESULT_DIR/"fleurs_vi_noise_top_errors.csv",index=False)
    display(pivot)
    best_clean=pivot.loc[pivot["Clean paired"].idxmin()]
    best_noisy=pivot.loc[pivot["0 dB"].idxmin()]
    best_drop=pivot.loc[pivot["wer_degradation_pp"].idxmin()]
    print(f"Best paired-clean ASR: {best_clean['model']} — {best_clean['Clean paired']:.2f}% WER")
    print(f"Best ASR at 0 dB: {best_noisy['model']} — {best_noisy['0 dB']:.2f}% WER")
    print(f"Lowest 0 dB degradation: {best_drop['model']} — {best_drop['wer_degradation_pp']:.2f} pp")
    ns.insert(0,"benchmark","FLEURS Vietnamese DEMAND 0 dB");ns["split"]="test"
    ALL_SUMMARIES.append(ns);ALL_DETAILS["fleurs_vi_noise"]=noise_det


if RUN["fleurs_en_clean"]:
    pred=run_predictions("fleurs_en_clean",FLEURS_EN_META,fleurs_en_wave,"en",DATASET_REVISIONS["google/fleurs"])
    det=detailed_metrics(FLEURS_EN_META,pred,normalize_en)
    fairness=pairwise_fairness(det,FLEURS_EN_META)
    audit=normalization_audit(FLEURS_EN_META,det)
    top=top_error_audit(det,10)
    summ=corpus_summary(det,["model"]);summ.insert(0,"benchmark","FLEURS English clean");summ["split"]="test";summ["condition"]="clean"
    det.to_csv(RESULT_DIR/"fleurs_en_clean_detailed.csv",index=False)
    summ.to_csv(RESULT_DIR/"fleurs_en_clean_summary.csv",index=False)
    fairness.to_csv(RESULT_DIR/"fleurs_en_clean_pairwise_fairness.csv",index=False)
    audit.to_csv(RESULT_DIR/"fleurs_en_clean_normalization_audit.csv",index=False)
    top.to_csv(RESULT_DIR/"fleurs_en_clean_top_errors.csv",index=False)
    FLEURS_EN_EXCLUDED.to_csv(RESULT_DIR/"fleurs_en_clean_excluded_samples.csv",index=False)
    ALL_DETAILS["fleurs_en_clean"]=det;ALL_SUMMARIES.append(summ)
    display(summ[["model","wer_percent","cer_percent","S","D","I","empty_predictions","estimated_device_rtf","estimated_device_x_realtime"]])
    display(top.head(min(30,len(top))))
    print_wer_ranking(summ,"English clean accuracy ranking")


VIMD_META=None
if RUN["vimd_regional"]:
    repo="nguyendv02/ViMD_Dataset";rev=DATASET_REVISIONS[repo]
    files=hf.list_repo_files(repo,repo_type="dataset",revision=rev)
    test_paths=sorted(x for x in files if re.fullmatch(r"data/test-\d+-of-\d+\.parquet",x))
    if not test_paths:raise RuntimeError("No physical ViMD test shards found")
    urls=[hf_hub_url(repo,x,repo_type="dataset",revision=rev) for x in test_paths]
    VIMD_DS=load_dataset("parquet",data_files={"test":urls},split="test")
    required={"text","region","speakerID","audio"}
    if not required.issubset(VIMD_DS.column_names):raise RuntimeError(f"Missing ViMD fields: {required-set(VIMD_DS.column_names)}")
    province_col=next((x for x in ["province","province_name","province_code"] if x in VIMD_DS.column_names),None)
    if province_col is None:raise RuntimeError("ViMD has no province metadata")
    if "set" in VIMD_DS.column_names:
        vals={str(x).strip().lower() for x in VIMD_DS.unique("set")}
        if vals!={"test"}:VIMD_DS=VIMD_DS.filter(lambda x:str(x["set"]).strip().lower()=="test")
    VIMD_DS=VIMD_DS.cast_column("audio",Audio(decode=False))
    rmap={"north":"North","central":"Central","south":"South"};valid=[];excluded=[];seen={x:0 for x in rmap.values()}
    if RUN_MODE=="smoke":
        region_targets={x:SMOKE_VIMD_PER_REGION for x in rmap.values()}
    elif RUN_MODE=="benchmark":
        region_targets=dict(VIMD_BENCHMARK_REGION_TARGETS)
    else:
        region_targets=None
    for idx,row in enumerate(tqdm(VIMD_DS,desc="Validate ViMD test")):
        filename=str(row.get("filename",idx));sid=f"{idx}::{filename}"
        try:
            region=rmap.get(str(row["region"]).strip().lower())
            if region is None:raise ValueError(f"bad region {row['region']}")
            if region_targets is not None and seen[region]>=region_targets[region]:continue
            ref=str(row["text"]);nr=normalize_vi(ref)
            if not nr:raise ValueError("empty normalized reference")
            x,sr=audio_array(row["audio"]);x,dur=validate_wave(x,sr)
            valid.append({"dataset_index":idx,"sample_key":f"vimd::{idx}::{filename}","sample_id":sid,"filename":filename,
                          "region":region,"province":str(row[province_col]),"speakerID":str(row["speakerID"]),
                          "duration_sec":dur,"raw_reference":ref,"normalized_reference":nr})
            seen[region]+=1
            if region_targets is not None and all(seen[r]>=region_targets[r] for r in region_targets):break
        except Exception as exc:excluded.append({"dataset_index":idx,"sample_id":sid,"reason":repr(exc)})
    VIMD_META=pd.DataFrame(valid)
    if set(VIMD_META.region)!={"North","Central","South"}:raise RuntimeError(f"ViMD selected set lacks a region: {VIMD_META.region.value_counts().to_dict()}")
    def vimd_wave(r):x,sr=audio_array(VIMD_DS[int(r.dataset_index)]["audio"]);return validate_wave(x,sr)[0]
    pred=run_predictions("vimd_regional",VIMD_META,vimd_wave,"vi",rev)
    det=detailed_metrics(VIMD_META,pred,normalize_vi)
    fairness=pairwise_fairness(det,VIMD_META)
    overall=corpus_summary(det,["model"]);overall["region"]="Overall"
    regional=corpus_summary(det,["model","region"])
    summary=pd.concat([overall,regional],ignore_index=True);summary.insert(0,"benchmark","ViMD regional accents");summary["split"]="test";summary["condition"]="clean"
    province=corpus_summary(det,["model","region","province"])
    wide=regional.pivot(index="model",columns="region",values="wer_percent").reindex(columns=["North","Central","South"])
    wide=wide.rename(columns={x:f"{x} WER (%) ↓" for x in wide.columns})
    reg_cols=["North WER (%) ↓","Central WER (%) ↓","South WER (%) ↓"]
    wide["Macro Regional WER (%) ↓"]=wide[reg_cols].mean(axis=1)
    wide["Accent Gap (pp) ↓"]=wide[reg_cols].max(axis=1)-wide[reg_cols].min(axis=1)
    wide["Worst-region WER (%) ↓"]=wide[reg_cols].max(axis=1)
    wide=wide.join(overall.set_index("model")["wer_percent"].rename("Overall WER (%) ↓")).reset_index().rename(columns={"model":"Model"})
    det.to_csv(RESULT_DIR/"vimd_detailed.csv",index=False)
    summary.to_csv(RESULT_DIR/"vimd_region_long_summary.csv",index=False)
    wide.to_csv(RESULT_DIR/"vimd_region_criterion_summary.csv",index=False)
    province.to_csv(RESULT_DIR/"vimd_province_diagnostic.csv",index=False)
    fairness.to_csv(RESULT_DIR/"vimd_pairwise_fairness.csv",index=False)
    pd.DataFrame(excluded).to_csv(RESULT_DIR/"vimd_excluded_samples.csv",index=False)
    top_error_audit(det,10).to_csv(RESULT_DIR/"vimd_top_errors.csv",index=False)
    ALL_DETAILS["vimd_regional"]=det;ALL_SUMMARIES.append(summary)
    display(wide)
    for label,col,unit in [("Best North",reg_cols[0],"% WER"),("Best Central",reg_cols[1],"% WER"),("Best South",reg_cols[2],"% WER"),
                           ("Lowest Macro Regional WER","Macro Regional WER (%) ↓","%"),("Smallest Accent Gap","Accent Gap (pp) ↓","pp"),
                           ("Lowest Worst-region WER","Worst-region WER (%) ↓","%")]:
        rr=wide.loc[wide[col].idxmin()];print(f"{label}: {rr['Model']} — {rr[col]:.2f} {unit}")


if RUN["vimedcss_codeswitch"]:
    repo="tensorxt/ViMedCSS";rev=DATASET_REVISIONS[repo]
    split_details=[];cs_audit=[];all_excluded=[];fairness_parts=[]
    splits=["test"] + (["hard"] if VIMEDCSS_INCLUDE_HARD else [])
    for split in splits:
        ds=load_dataset(repo,split=split,revision=rev)
        needed={"segment_id","segment_text","cs_terms_list","audio"}
        if not needed.issubset(ds.column_names):raise RuntimeError(f"ViMedCSS {split} missing {needed-set(ds.column_names)}")
        ds=ds.cast_column("audio",Audio(decode=False))
        valid=[];excluded=[];limit=SMOKE_N if RUN_MODE=="smoke" else (BENCHMARK_N if RUN_MODE=="benchmark" else None)
        for idx,row in enumerate(tqdm(ds,desc=f"Validate ViMedCSS {split}")):
            sid=str(row["segment_id"])
            try:
                ref=str(row["segment_text"]);nr=normalize_vi(ref)
                if not nr:raise ValueError("empty normalized reference")
                x,sr=audio_array(row["audio"]);x,dur=validate_wave(x,sr)
                mask,failures=align_cs_terms_to_reference(nr,row.get("cs_terms_list"))
                if not any(mask): failures=failures or ["NO_ALIGNED_CS_TOKEN"]
                valid.append({"dataset_index":idx,"sample_key":f"vimedcss::{split}::{sid}","sample_id":sid,"split":split,
                              "duration_sec":dur,"raw_reference":ref,"normalized_reference":nr,"cs_terms_list":str(row.get("cs_terms_list") or ""),
                              "cs_terms_count":int(row.get("cs_terms_count") or len(parse_cs_terms(row.get("cs_terms_list")))),
                              "topic":str(row.get("topic",row.get("Topic",""))),"cs_mask_json":json.dumps(mask),
                              "cs_alignment_ok":not failures,"cs_alignment_failures":json.dumps(failures,ensure_ascii=False)})
                if limit and len(valid)>=limit:break
            except Exception as exc:excluded.append({"dataset_index":idx,"sample_id":sid,"reason":repr(exc)})
        meta=pd.DataFrame(valid)
        if excluded:
            ex=pd.DataFrame(excluded);ex["split"]=split;all_excluded.append(ex)
        if meta.empty:raise RuntimeError(f"No valid ViMedCSS {split} samples")
        def wave(r,ds=ds):x,sr=audio_array(ds[int(r.dataset_index)]["audio"]);return validate_wave(x,sr)[0]
        pred=run_predictions(f"vimedcss_{split}",meta,wave,"vi",rev)
        det=detailed_metrics(meta,pred,normalize_vi)
        fair=pairwise_fairness(det,meta);fair["split"]=split;fairness_parts.append(fair)
        cs_rows=[]
        for r in det.itertuples():
            if bool(r.cs_alignment_ok):cs_rows.append(projected_cs_metrics(r.normalized_reference,r.normalized_prediction,json.loads(r.cs_mask_json)))
            else:cs_rows.append({k:np.nan for k in ["cs_hits","cs_S","cs_D","cs_I","cs_N","cs_wer","cs_token_accuracy","n_hits","n_S","n_D","n_I","n_N","n_wer","n_token_accuracy"]})
        det=pd.concat([det.reset_index(drop=True),pd.DataFrame(cs_rows)],axis=1)
        det.to_csv(RESULT_DIR/f"vimedcss_{split}_detailed.csv",index=False)
        split_details.append(det)
        cs_audit.append(meta[["sample_key","sample_id","split","cs_terms_list","cs_alignment_ok","cs_alignment_failures"]])

    VIMED_DET=pd.concat(split_details,ignore_index=True)
    VIMED_AUDIT=pd.concat(cs_audit,ignore_index=True);VIMED_AUDIT.to_csv(RESULT_DIR/"vimedcss_cs_alignment_audit.csv",index=False)
    if fairness_parts:pd.concat(fairness_parts,ignore_index=True).to_csv(RESULT_DIR/"vimedcss_pairwise_fairness.csv",index=False)
    if all_excluded:pd.concat(all_excluded,ignore_index=True).to_csv(RESULT_DIR/"vimedcss_excluded_samples.csv",index=False)
    else:pd.DataFrame(columns=["dataset_index","sample_id","reason","split"]).to_csv(RESULT_DIR/"vimedcss_excluded_samples.csv",index=False)
    top_error_audit(VIMED_DET,10).to_csv(RESULT_DIR/"vimedcss_top_errors.csv",index=False)
    rows=[]
    for (model,split),g in VIMED_DET.groupby(["model","split"],sort=False):
        base=corpus_summary(g,[]).iloc[0].to_dict()
        eligible=g[g.cs_alignment_ok.astype(bool)].copy()
        csN=int(eligible.cs_N.sum());nN=int(eligible.n_N.sum())
        cs_err=float(eligible.cs_S.sum()+eligible.cs_D.sum()+eligible.cs_I.sum());n_err=float(eligible.n_S.sum()+eligible.n_D.sum()+eligible.n_I.sum())
        cs_hits=float(eligible.cs_hits.sum())
        base.update({"benchmark":"ViMedCSS code-switch","model":model,"split":split,"condition":"clean",
                     "cs_metric_samples":len(eligible),"cs_alignment_excluded":len(g)-len(eligible),"cs_reference_tokens":csN,
                     "cs_wer":cs_err/csN if csN else np.nan,"cs_wer_percent":100*cs_err/csN if csN else np.nan,
                     "cs_token_accuracy":cs_hits/csN if csN else np.nan,"cs_token_accuracy_percent":100*cs_hits/csN if csN else np.nan,
                     "n_reference_tokens":nN,"n_wer":n_err/nN if nN else np.nan,"n_wer_percent":100*n_err/nN if nN else np.nan})
        rows.append(base)
    VIMED_SUM=pd.DataFrame(rows)
    VIMED_SUM.to_csv(RESULT_DIR/"vimedcss_summary.csv",index=False);ALL_SUMMARIES.append(VIMED_SUM);ALL_DETAILS["vimedcss"]=VIMED_DET
    display(VIMED_SUM[["model","split","wer_percent","cer_percent","cs_wer_percent","cs_token_accuracy_percent","n_wer_percent","cs_alignment_excluded","estimated_device_rtf"]])


MODEL_METADATA_DF=pd.DataFrame([{
    "model":label,"hf_repo":MODEL_IDS[label],"hf_revision":MODEL_REVISIONS[MODEL_IDS[label]],
    "artifact_mode":a["artifact_mode"],"linked_model_id":a.get("linked_model_id"),
    "encoder_model_id":a["encoder_model"].model_id,"decoder_model_id":a["decoder_model"].model_id,
    "producer_device":a.get("producer_device"),"encoder_model_type":a.get("encoder_model_type"),
    "decoder_model_type":a.get("decoder_model_type"),"qairt_api_version":QAIRT_VERSION,"qairt_full_version":QAIRT_FULL_VERSION,
    "encoder_graph":a["graph_names"][0],"decoder_graph":a["graph_names"][1],
} for label,a in MODEL_ARTIFACTS.items()])
MODEL_METADATA_DF.to_csv(RESULT_DIR/"MODEL_METADATA.csv",index=False)

if ALL_SUMMARIES:
    consolidated=pd.concat(ALL_SUMMARIES,ignore_index=True,sort=False)
    consolidated["run_mode"]=RUN_MODE
    consolidated=consolidated.merge(PROFILE_DF[["model","encoder_ms","decoder_ms_per_token","encoder_plus_first_decoder_ms","peak_ram_mb"]],on="model",how="left")
    consolidated.to_csv(RESULT_DIR/"ALL_BENCHMARK_SUMMARIES.csv",index=False)
    display(consolidated)

# ---------- ONE compact final table ----------
def _series_from(path, model_col, value_col, extra_filter=None):
    df=pd.read_csv(path)
    if extra_filter is not None:
        df=extra_filter(df)
    return df.set_index(model_col)[value_col]

final=pd.DataFrame({"Model":list(MODEL_IDS.keys())}).set_index("Model")
final["VN WER (%) ↓"]=_series_from(RESULT_DIR/"fleurs_vi_clean_summary.csv","model","wer_percent")
final["EN WER (%) ↓"]=_series_from(RESULT_DIR/"fleurs_en_clean_summary.csv","model","wer_percent")
reg=pd.read_csv(RESULT_DIR/"vimd_region_criterion_summary.csv").set_index("Model")
for c in ["North WER (%) ↓","Central WER (%) ↓","South WER (%) ↓"]:
    final[c]=reg[c]
noise=pd.read_csv(RESULT_DIR/"fleurs_vi_noise_robustness_summary.csv").set_index("model")
final["Noise ΔWER @0dB (pp) ↓"]=noise["wer_degradation_pp"]
cs=pd.read_csv(RESULT_DIR/"vimedcss_summary.csv")
cs=cs[cs["split"].astype(str)=="test"].set_index("model")
final["Code-switch CS-WER (%) ↓"]=cs["cs_wer_percent"]
prof=PROFILE_DF.set_index("model")
final["S24 Encoder (ms) ↓"]=prof["encoder_ms"]
final["S24 Decoder/token (ms) ↓"]=prof["decoder_ms_per_token"]
final["Peak RAM on S24 (MB) ↓"]=prof["peak_ram_mb"]
final=final.reset_index()

# Human-readable rounding; raw detailed files remain available in results/.
for c in final.columns:
    if c!="Model":final[c]=pd.to_numeric(final[c],errors="coerce").round(2)
FINAL_REQUIRED_FIELDS=[
    "VN WER (%) ↓",
    "EN WER (%) ↓",
    "North WER (%) ↓",
    "Central WER (%) ↓",
    "South WER (%) ↓",
    "Noise ΔWER @0dB (pp) ↓",
    "Code-switch CS-WER (%) ↓",
    "S24 Encoder (ms) ↓",
    "S24 Decoder/token (ms) ↓",
    "Peak RAM on S24 (MB) ↓",
]
_validate_required_final_rows(final.to_dict("records"),list(MODEL_IDS),FINAL_REQUIRED_FIELDS)
FINAL_TABLE_PATH=RESULT_DIR/"FINAL_BENCHMARK_TABLE.csv"
final.to_csv(FINAL_TABLE_PATH,index=False)
# Convenience copy: one obvious file at WORK_ROOT for Drive upload / submission.
FINAL_UPLOAD_PATH=WORK_ROOT/"FINAL_BENCHMARK_TABLE.csv"
shutil.copy2(FINAL_TABLE_PATH,FINAL_UPLOAD_PATH)
print("\nFINAL BENCHMARK TABLE")
display(final)
print("\nONE-FILE OUTPUT:",FINAL_UPLOAD_PATH)

inventory={
    **BASE_MANIFEST,"run_mode":RUN_MODE,"run_flags":RUN,"hub_microbatch":HUB_MICROBATCH,
    "benchmark_sampling":{
        "fleurs_vi": len(FLEURS_VI_META),
        "fleurs_en": len(FLEURS_EN_META),
        "vimd_total": len(VIMD_META) if VIMD_META is not None else 0,
        "vimd_regions": (VIMD_META.region.value_counts().to_dict() if VIMD_META is not None else {}),
        "vimedcss_test": int((VIMED_DET.split.astype(str)=="test").sum()/len(MODEL_IDS)) if "VIMED_DET" in globals() else 0,
    },
    "optimized_artifacts":{label:{
        "artifact_mode":a["artifact_mode"],"linked_model_id":a.get("linked_model_id"),
        "encoder_model_id":a["encoder_model"].model_id,"decoder_model_id":a["decoder_model"].model_id,
        "graphs":a["graph_names"],
    } for label,a in MODEL_ARTIFACTS.items()},
    "profile_rows":PROFILE_DF.to_dict("records"),
    "final_table":str(FINAL_UPLOAD_PATH),
    "result_dir":str(RESULT_DIR),
    "cs_wer_method":{
        "annotation_field":"cs_terms_list","reference_alignment":"exact contiguous match after normalize_vi",
        "edit_alignment":"full-sentence Levenshtein; S/D to reference class",
        "insertion_rule":"CS iff insertion boundary touches at least one CS reference token; otherwise N",
        "unalignable_annotation":"kept in overall WER/CER; excluded from CS-WER and N-WER",
        "cs_token_accuracy":"CS hits / CS reference tokens",
    },
}
atomic_json(inventory,RESULT_DIR/"REPRODUCIBILITY_INVENTORY.json")

# ---------- Submission/evidence bundle: one ZIP to upload ----------
def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

# Convert the append-only Qualcomm inference ledger into a compact CSV.
ledger_records=[]
if JOB_LEDGER_JSONL.exists():
    for line in JOB_LEDGER_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: ledger_records.append(json.loads(line))
            except Exception: pass
ledger_df=pd.DataFrame(ledger_records)
if not ledger_df.empty and "job_id" in ledger_df:
    ledger_df=ledger_df.drop_duplicates(subset=["job_id"],keep="last").reset_index(drop=True)
JOB_LEDGER_CSV=RESULT_DIR/"QUALCOMM_JOB_LEDGER.csv"
ledger_df.to_csv(JOB_LEDGER_CSV,index=False)
if not ledger_df.empty and "device" in ledger_df:
    seen_devices=set(ledger_df["device"].dropna().astype(str))
    if seen_devices != {TARGET_DEVICE_NAME}:
        raise RuntimeError(f"Evidence ledger contains non-target device(s): {sorted(seen_devices)}")

# Per-sample proof survives resumed runs because checkpoints persist after each completed microbatch.
proof_parts=[]
for cp in sorted(CHECKPOINT_DIR.glob("*.csv")):
    try:
        df=pd.read_csv(cp,keep_default_na=False)
    except Exception:
        continue
    needed=[c for c in ["sample_key","model","artifact_mode","encoder_model_id","decoder_model_id","linked_model_id",
                         "encoder_job_id","last_decoder_job_id","decoder_job_count","decode_steps","truncated"] if c in df.columns]
    if needed:
        part=df[needed].copy()
        part.insert(0,"checkpoint_file",cp.name)
        proof_parts.append(part)
sample_proof=pd.concat(proof_parts,ignore_index=True,sort=False) if proof_parts else pd.DataFrame()
SAMPLE_JOB_PROOF=RESULT_DIR/"INFERENCE_SAMPLE_JOB_PROOF.csv"
sample_proof.to_csv(SAMPLE_JOB_PROOF,index=False)

# Compact exact sample selection used by each benchmark.
selection_parts=[]
for bench,key in ALL_DETAILS.items():
    cols=[c for c in ["sample_key","sample_id","dataset_index","split","region","condition","duration_sec","raw_reference","normalized_reference"] if c in key.columns]
    if cols:
        dedupe_cols=["sample_key","condition"] if "condition" in cols else ["sample_key"]
        x=key[cols].drop_duplicates(subset=dedupe_cols).copy()
        x.insert(0,"benchmark",bench)
        selection_parts.append(x)
sample_selection=pd.concat(selection_parts,ignore_index=True,sort=False) if selection_parts else pd.DataFrame()
SAMPLE_SELECTION=RESULT_DIR/"SAMPLE_SELECTION.csv"
sample_selection.to_csv(SAMPLE_SELECTION,index=False)

if RUN_MODE=="benchmark":
    if len(FLEURS_VI_META)!=BENCHMARK_N or len(FLEURS_EN_META)!=BENCHMARK_N:
        raise RuntimeError(f"Benchmark sampling contract failed for FLEURS: vi={len(FLEURS_VI_META)}, en={len(FLEURS_EN_META)}")
    if VIMD_META is None or len(VIMD_META)!=BENCHMARK_N:
        raise RuntimeError(f"Benchmark sampling contract failed for ViMD: {0 if VIMD_META is None else len(VIMD_META)}")
    if "VIMED_DET" not in globals() or VIMED_DET[VIMED_DET.split.astype(str)=="test"].sample_key.nunique()!=BENCHMARK_N:
        got=0 if "VIMED_DET" not in globals() else int(VIMED_DET[VIMED_DET.split.astype(str)=="test"].sample_key.nunique())
        raise RuntimeError(f"Benchmark sampling contract failed for ViMedCSS test: {got}")

run_finished_utc=datetime.now(timezone.utc).isoformat()
compile_link_evidence={}
for label,a in MODEL_ARTIFACTS.items():
    compile_link_evidence[label]={
        "artifact_mode":a.get("artifact_mode"),
        "linked_model_id":a.get("linked_model_id"),
        "linked_model_type":a.get("linked_model_type"),
        "encoder_model_id":a.get("encoder_model_id"),
        "decoder_model_id":a.get("decoder_model_id"),
        "encoder_model_type":a.get("encoder_model_type"),
        "decoder_model_type":a.get("decoder_model_type"),
        "producer_device":a.get("producer_device"),
        "compile_job_ids":a.get("compile_job_ids",[]),
        "compile_job_urls":a.get("compile_job_urls",[]),
        "link_job_id":a.get("link_job_id"),
        "link_job_url":a.get("link_job_url"),
        "initial_link_job_id":a.get("initial_link_job_id"),
        "initial_link_job_url":a.get("initial_link_job_url"),
        "link_attempts":a.get("link_attempts",[]),
        "graphs":a.get("graph_names",[]),
    }

representative_jobs=[]
if not ledger_df.empty:
    keep=[c for c in ["job_id","job_url","job_name","device","artifact_mode","component","artifact_model_id",
                      "linked_model_id","graph","batch_size","compute_unit_requested","qairt_api_version"] if c in ledger_df.columns]
    if keep:
        rep=pd.concat([ledger_df.head(5),ledger_df.tail(5)],ignore_index=True).drop_duplicates(subset=["job_id"])
        representative_jobs=rep[keep].to_dict("records")

evidence={
    "completed_successfully": True,
    "run_started_utc": RUN_STARTED_UTC,
    "run_finished_utc": run_finished_utc,
    "claim": "All Whisper encoder/decoder neural forward passes used for predictions were submitted through Qualcomm AI Hub Workbench to the exact hosted Samsung Galaxy S24; NPU compute was explicitly requested.",
    "important_scope_note": "The Windows PC performs dataset download/audio preprocessing, API orchestration, token decoding, and metric calculation. Neural encoder/decoder forward computation is remote on the Qualcomm-hosted S24.",
    "exact_target_device": TARGET_DEVICE_NAME,
    "device_fingerprint": DEVICE_FINGERPRINT,
    "device_details": DEVICE_FINGERPRINT_PAYLOAD,
    "qairt_api_version": QAIRT_VERSION,
    "qairt_full_version": QAIRT_FULL_VERSION,
    "compute_unit_requested": "npu",
    "run_mode": RUN_MODE,
    "benchmark_n": BENCHMARK_N,
    "hub_microbatch": HUB_MICROBATCH,
    "dataset_revisions": DATASET_REVISIONS,
    "model_revisions": MODEL_REVISIONS,
    "compile_link": compile_link_evidence,
    "profiles": PROFILE_DF.to_dict("records"),
    "inference_ledger": {
        "unique_successful_inference_jobs_recorded": int(len(ledger_df)),
        "sample_executions_inside_jobs": int(pd.to_numeric(ledger_df.get("batch_size",pd.Series(dtype=float)),errors="coerce").fillna(0).sum()) if not ledger_df.empty else 0,
        "devices_seen": sorted(set(ledger_df["device"].dropna().astype(str))) if (not ledger_df.empty and "device" in ledger_df) else [],
        "representative_jobs": representative_jobs,
        "ledger_csv": JOB_LEDGER_CSV.name,
    },
    "per_sample_job_proof_csv": SAMPLE_JOB_PROOF.name,
    "sample_selection_csv": SAMPLE_SELECTION.name,
    "final_table_sha256": _sha256_file(FINAL_UPLOAD_PATH),
    "final_table": FINAL_UPLOAD_PATH.name,
    "api_token_stored": False,
    "auth_mode": "session_ClientConfig",
}
RUN_EVIDENCE=RESULT_DIR/"RUN_EVIDENCE.json"
atomic_json(evidence,RUN_EVIDENCE)

readme_evidence=RESULT_DIR/"README_EVIDENCE.txt"
readme_evidence.write_text(
    "QUALCOMM S24 BENCHMARK EVIDENCE\n\n"
    "1) FINAL_BENCHMARK_TABLE.csv: final metrics.\n"
    "2) RUN_EVIDENCE.json: exact S24 device fingerprint, model/dataset revisions, optimized artifact IDs, compile/link/profile job IDs and representative inference job URLs.\n"
    "3) QUALCOMM_JOB_LEDGER.csv: every successful Qualcomm inference job recorded during this run, including job ID/URL, exact device, graph, artifact model and batch size.\n"
    "4) INFERENCE_SAMPLE_JOB_PROOF.csv: per-sample encoder and final decoder Qualcomm job IDs from resumable checkpoints.\n"
    "5) SAMPLE_SELECTION.csv: exact fixed samples used for evaluation.\n"
    "6) MODEL_METADATA.csv and s24_model_speed_memory_profile.csv: linked-context or separate-QNN-DLC artifacts and S24 profile evidence.\n"
    "7) RUN_CONSOLE.log: local console audit log (API token is never echoed or stored).\n\n"
    "Open the Qualcomm job URLs while signed into the team's AI Hub account to independently inspect the jobs.\n",
    encoding="utf-8"
)

bundle_files=[
    FINAL_UPLOAD_PATH,
    RUN_EVIDENCE,
    JOB_LEDGER_CSV,
    SAMPLE_JOB_PROOF,
    SAMPLE_SELECTION,
    RESULT_DIR/"MODEL_METADATA.csv",
    RESULT_DIR/"s24_model_speed_memory_profile.csv",
    RESULT_DIR/"run_manifest.json",
    RESULT_DIR/"REPRODUCIBILITY_INVENTORY.json",
    readme_evidence,
    RUN_CONSOLE_LOG,
]
bundle_files=[Path(x) for x in bundle_files if Path(x).exists()]
SHA_PATH=RESULT_DIR/"SHA256SUMS.txt"
SHA_PATH.write_text("\n".join(f"{_sha256_file(x)}  {x.name}" for x in bundle_files)+"\n",encoding="utf-8")
bundle_files.append(SHA_PATH)

BUNDLE_PATH=WORK_ROOT/"QAI_S24_BENCHMARK_SUBMISSION.zip"
tmp_bundle=BUNDLE_PATH.with_suffix(".zip.tmp")
if tmp_bundle.exists(): tmp_bundle.unlink()
with zipfile.ZipFile(tmp_bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
    for x in bundle_files:
        zf.write(x,arcname=x.name)
tmp_bundle.replace(BUNDLE_PATH)

print("\nRESULTS:",RESULT_DIR)
print("Compiled artifact cache:",REMOTE_MANIFEST_PATH)
print("Profile cache:",PROFILE_PATH)
print("\nFINAL ONE-FILE SUBMISSION BUNDLE:",BUNDLE_PATH)
print("Contains final table + Qualcomm job IDs/URLs + exact S24 device evidence + hashes.")
if RUN_MODE=="smoke":
    print("Smoke passed. Set QAI_RUN_MODE=benchmark for the requested 50-sample benchmark.")
