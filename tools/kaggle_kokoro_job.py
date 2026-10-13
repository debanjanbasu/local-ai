"""Stage (never submit) private Kaggle CPU jobs for the Kokoro-82M speech model.

`--kernel ... --mode reference` stages a CPU job with internet that installs the
pinned `kokoro` and `misaki` packages, downloads `hexgrad/Kokoro-82M`
(Apache-2.0) with its upstream file metadata, and writes one archive
(`/kaggle/working/kokoro-ref.tar`) holding:

- `model/`: `config.json`, `kokoro-v1_0.pth`, the English voice packs and the
  model card, unchanged (`tools/convert_kokoro.py` checks them against pins);
- `src/`: the exact `kokoro` and `misaki` sources and lexicons that ran;
- `fixtures/`: parity fixtures for `local-engine` (phonemes, token ids,
  durations and waveforms, F32 little-endian) for every row of `FIXTURES`, run
  twice: with the source module's random phase and noise replaced by zeros
  (deterministic, for waveform parity) and seeded (as users hear it), plus
  full `KPipeline` outputs from the original G2P, intermediate activations for
  the first rows, and `g2p.jsonl` (misaki phonemes for the fixture texts and a
  public-domain Gutenberg sample);
- `asr.json`: a reference ASR transcript of each pipeline output.

`--mode asr --dataset owner/slug` stages a job that transcribes every
`*.wav` under the attached private dataset with the same ASR model and writes
`/kaggle/working/asr.json` (used to compare native outputs with the
reference by word error rate).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

KOKORO_VERSION = "0.9.4"
MISAKI_VERSION = "0.9.4"
REPO = "hexgrad/Kokoro-82M"
ASR_MODEL = "openai/whisper-large-v3-turbo"
UV_VERSION = "0.8.22"
TORCH_CPU = "torch==2.8.0+cpu"
ENGLISH_VOICES = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
# (text, voice, speed). Fixed before any native output existed.
FIXTURES = (
    ("The quick brown fox jumps over the lazy dog.", "af_heart", 1.0),
    ("Hello there! How are you doing today?", "af_bella", 1.0),
    ("She sells seashells by the seashore.", "am_michael", 1.0),
    ("Please call Stella and ask her to bring these things with her from the store.",
     "bf_emma", 1.0),
    ("I would like a cup of tea, please.", "bm_george", 1.0),
    ("Where were you on the night of the fourteenth?", "af_alloy", 1.0),
    ("Our meeting has been moved to three thirty in the afternoon.", "am_echo", 1.0),
    ("Never underestimate the power of a good night's sleep.", "bm_fable", 1.0),
    ("The museum opens at nine and closes at five.", "am_onyx", 1.0),
    ("Can you believe it? We actually won the championship!", "af_nova", 1.0),
    ("Machine learning models can now generate remarkably natural speech.",
     "af_sarah", 1.0),
    ("Turn left at the second traffic light, then continue straight for two miles.",
     "af_sky", 1.0),
    ("Yes.", "af_heart", 1.0),
    ("No, I don't think that's a good idea.", "am_puck", 1.0),
    ("The weather forecast predicts heavy rain tomorrow morning.", "am_fenrir", 1.0),
    ("He read the book that she had read the year before.", "af_jessica", 1.0),
    ("Learning a new language takes patience, practice, and a little courage.",
     "af_heart", 0.8),
    ("Quickly, quickly, the train is about to leave the station!", "am_adam", 1.25),
    ("The colour of the autumn leaves was absolutely beautiful.", "bf_isabella", 1.0),
    ("I have been waiting for this moment for a very long time.", "bm_lewis", 1.0),
    ("Every scientist knows that careful measurement is the foundation of discovery.",
     "am_eric", 1.0),
    ("Would you rather explore the deep ocean or travel to distant planets?",
     "af_kore", 1.0),
    ("The children laughed and played in the garden until the sun went down.",
     "af_river", 1.0),
    ("Thank you for calling. Your call is important to us.", "af_nicole", 1.0),
    ("A journey of a thousand miles begins with a single step.", "bf_lily", 1.0),
    ("Bright stars shimmered quietly above the silent mountain lake.", "bm_daniel", 1.0),
    ("Software engineers often debate the merits of different programming styles,"
     " but most agree that clear and simple code is easier to maintain over time.",
     "af_heart", 1.0),
    ("Despite the storm, the small fishing boat returned safely to the harbour just"
     " before midnight, and the whole village gathered on the pier to welcome the"
     " exhausted crew home.", "am_michael", 1.0),
    ("It's not what you look at that matters, it's what you see.", "bf_alice", 1.0),
    ("The library will be closed on Monday for maintenance.", "am_liam", 1.0),
)
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
GUTENBERG_SENTENCES = 400
INTERMEDIATE_ROWS = 3

REFERENCE_JOB = r"""
import hashlib, io, json, os, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path

def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)

# kokoro/misaki 0.9.4 need Python < 3.13 (Kaggle runs 3.13): rerun this script
# in a seeded Python 3.12 virtual environment with CPU torch.
if not os.environ.get("KOKORO_JOB_VENV"):
    run(sys.executable, "-m", "pip", "install", "-q", "uv==" + UV_VERSION)
    run("uv", "venv", "--seed", "-p", "3.12", "/kaggle/tmp/venv")
    python = "/kaggle/tmp/venv/bin/python"
    run("uv", "pip", "install", "--python", python, TORCH_CPU, "--index-url",
        "https://download.pytorch.org/whl/cpu")
    run("uv", "pip", "install", "--python", python, "kokoro==" + KOKORO_VERSION,
        "misaki[en]==" + MISAKI_VERSION, "jiwer", "soundfile")
    os.execve(python, [python, os.path.abspath(__file__)], {**os.environ, "KOKORO_JOB_VENV": "1"})

import numpy as np
import torch
import huggingface_hub
import kokoro, misaki

work = Path("/kaggle/working/kokoro-ref")
work.mkdir(parents=True)
torch.set_num_threads(4)

api = huggingface_hub.HfApi()
info = api.model_info(REPO, files_metadata=True)
files = {s.rfilename: {"size": s.size, "lfs_sha256": (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)} for s in info.siblings}
wanted = ["config.json", "kokoro-v1_0.pth", "README.md", "VOICES.md"] + ["voices/" + v + ".pt" for v in ENGLISH_VOICES]
model_dir = work / "model"
for name in wanted:
    path = huggingface_hub.hf_hub_download(REPO, name, revision=info.sha)
    target = model_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
digests = {name: {"bytes": (model_dir / name).stat().st_size, "sha256": hashlib.sha256((model_dir / name).read_bytes()).hexdigest()} for name in wanted}
for name, d in digests.items():
    up = files.get(name, {})
    if up.get("lfs_sha256") and up["lfs_sha256"] != d["sha256"]:
        raise RuntimeError("upstream LFS sha256 mismatch for " + name)
(work / "upstream.json").write_text(json.dumps({"repo": REPO, "revision": info.sha, "files": digests, "upstream": {k: files.get(k) for k in wanted}, "versions": {"kokoro": kokoro.__version__ if hasattr(kokoro, "__version__") else KOKORO_VERSION, "misaki": MISAKI_VERSION, "torch": torch.__version__, "python": sys.version}}, indent=1))

src = work / "src"
for package in (kokoro, misaki):
    root = Path(package.__file__).parent
    shutil.copytree(root, src / root.name, ignore=shutil.ignore_patterns("__pycache__"))

from kokoro import KModel, KPipeline
from misaki import en
model = KModel(repo_id=REPO, config=str(model_dir / "config.json"), model=str(model_dir / "kokoro-v1_0.pth")).eval()
pipes = {lang: KPipeline(lang_code=lang, repo_id=REPO, model=model) for lang in "ab"}
g2p_plain = {lang: en.G2P(trf=False, british=(lang == "b"), fallback=None) for lang in "ab"}
packs = {v: torch.load(model_dir / "voices" / (v + ".pt"), weights_only=True) for v in ENGLISH_VOICES}

fixtures = work / "fixtures"
for sub in ("zero", "seed", "pipeline", "intermediates"):
    (fixtures / sub).mkdir(parents=True)

class ZeroNoise:
    def __enter__(self):
        self.saved = (torch.rand, torch.randn_like, torch.randn, torch.rand_like)
        def zeros(*shape, **kw):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                shape = tuple(shape[0])
            return torch.zeros(*shape, device=kw.get("device"), dtype=kw.get("dtype") or torch.float32)
        torch.rand = zeros
        torch.randn = zeros
        torch.randn_like = lambda t, **kw: torch.zeros_like(t)
        torch.rand_like = lambda t, **kw: torch.zeros_like(t)
    def __exit__(self, *exc):
        torch.rand, torch.randn_like, torch.randn, torch.rand_like = self.saved

def capture(row_dir):
    acts, hooks = {}, []
    names = dict(model.named_modules())
    keep = ["bert", "bert_encoder", "predictor.text_encoder", "predictor.lstm", "predictor.duration_proj",
            "predictor.shared", "predictor.F0_proj", "predictor.N_proj", "text_encoder",
            "decoder.encode", "decoder.decode.0", "decoder.decode.3", "decoder.asr_res",
            "decoder.generator.m_source", "decoder.generator.noise_convs.0", "decoder.generator.noise_res.0",
            "decoder.generator.ups.0", "decoder.generator.resblocks.0", "decoder.generator.noise_convs.1",
            "decoder.generator.noise_res.1", "decoder.generator.ups.1", "decoder.generator.conv_post",
            "decoder.generator"]
    for name in keep:
        module = names[name]
        def hook(mod, inputs, output, name=name):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            if torch.is_tensor(out):
                acts[name] = out.detach().float().numpy()
            if name == "decoder.generator":
                acts["decoder.generator.input"] = inputs[0].detach().float().numpy()
                acts["decoder.generator.f0"] = inputs[2].detach().float().numpy()
        hooks.append(module.register_forward_hook(hook))
    return acts, hooks

rows = []
for index, (text, voice, speed) in enumerate(FIXTURES):
    lang = voice[0]
    phonemes, _ = g2p_plain[lang](text)
    pipe_chunks = list(pipes[lang](text, voice=voice, speed=speed))
    pipe_phonemes = [r.phonemes for r in pipe_chunks]
    pipe_audio = np.concatenate([r.audio.numpy() for r in pipe_chunks]).astype("<f4")
    (fixtures / "pipeline" / f"{index:03d}.f32").write_bytes(pipe_audio.tobytes())
    ps = pipe_phonemes[0] if len(pipe_phonemes) == 1 else None
    if ps is None:
        raise RuntimeError("fixture spans several pipeline chunks: " + text)
    ids = [model.vocab[p] for p in ps if p in model.vocab]
    ref_s = packs[voice][len(ps) - 1]
    acts, hooks = capture(fixtures) if index < INTERMEDIATE_ROWS else ({}, [])
    with ZeroNoise():
        out = model(ps, ref_s, speed, return_output=True)
    for h in hooks:
        h.remove()
    if acts:
        np.savez(fixtures / "intermediates" / f"{index:03d}.npz", **acts)
    zero = out.audio.numpy().astype("<f4")
    (fixtures / "zero" / f"{index:03d}.f32").write_bytes(zero.tobytes())
    torch.manual_seed(index)
    seeded = model(ps, ref_s, speed, return_output=True)
    (fixtures / "seed" / f"{index:03d}.f32").write_bytes(seeded.audio.numpy().astype("<f4").tobytes())
    rows.append({"index": index, "text": text, "voice": voice, "speed": speed, "lang": lang,
                 "phonemes_plain": phonemes, "phonemes": ps, "input_ids": [0, *ids, 0],
                 "pred_dur": out.pred_dur.tolist(), "seed_pred_dur": seeded.pred_dur.tolist(),
                 "samples": int(zero.shape[0]), "pipeline_samples": int(pipe_audio.shape[0])})
    print(index, text, ps, flush=True)
(fixtures / "manifest.json").write_text(json.dumps({"sample_rate": 24000, "rows": rows}, indent=1, ensure_ascii=False))

g2p_rows = [{"source": "fixture", "lang": lang, "text": t, "phonemes": g2p_plain[lang](t)[0]} for t, _, _ in FIXTURES for lang in "ab"]
try:
    raw = urllib.request.urlopen(GUTENBERG_URL, timeout=60).read().decode("utf-8", "replace")
    body = raw.split("*** START OF", 1)[-1].split("*** END OF", 1)[0]
    import re
    sentences = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+", body)]
    sentences = [s for s in sentences if 20 <= len(s) <= 300 and not s.isupper()][200:200 + GUTENBERG_SENTENCES]
    for s in sentences:
        for lang in "ab":
            g2p_rows.append({"source": "gutenberg-1342", "lang": lang, "text": s, "phonemes": g2p_plain[lang](s)[0]})
except Exception as error:
    print("gutenberg sample skipped:", error, flush=True)
with open(work / "g2p.jsonl", "w", encoding="utf-8") as f:
    for r in g2p_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from transformers import pipeline as hf_pipeline
asr = hf_pipeline("automatic-speech-recognition", model=ASR_MODEL, device="cpu")
normalize = asr.tokenizer.normalize
import jiwer
transcripts = {}
for r in rows:
    audio = np.frombuffer((fixtures / "pipeline" / f"{r['index']:03d}.f32").read_bytes(), dtype="<f4")
    text = asr({"raw": audio.astype(np.float32), "sampling_rate": 24000}, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
    transcripts[f"{r['index']:03d}"] = {"reference_text": r["text"], "asr": text}
refs = [normalize(v["reference_text"]) for v in transcripts.values()]
hyps = [normalize(v["asr"]) for v in transcripts.values()]
(work / "asr.json").write_text(json.dumps({"model": ASR_MODEL, "wer": jiwer.wer(refs, hyps), "rows": transcripts}, indent=1))
print("reference pipeline WER", jiwer.wer(refs, hyps), flush=True)

with tarfile.open("/kaggle/working/kokoro-ref.tar", "w") as tar:
    tar.add(work, arcname="kokoro-ref")
shutil.rmtree(work)
print(hashlib.sha256(Path("/kaggle/working/kokoro-ref.tar").read_bytes()).hexdigest())
"""

ASR_JOB = r"""
import json, subprocess, sys, wave
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], check=True)
import numpy as np, jiwer
from transformers import pipeline as hf_pipeline
owner, slug = DATASET.split("/")
roots = [p for p in (Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug) if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError("dataset not found")
texts = json.loads(next(roots[0].rglob("texts.json")).read_text())
asr = hf_pipeline("automatic-speech-recognition", model=ASR_MODEL, device="cpu")
normalize = asr.tokenizer.normalize
out = {"model": ASR_MODEL, "sets": {}}
for path in sorted(roots[0].rglob("*.wav")):
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    text = asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
    group = path.parent.name
    out["sets"].setdefault(group, {})[path.stem] = {"reference_text": texts[path.stem], "asr": text}
    print(group, path.stem, text, flush=True)
for group, rows in out["sets"].items():
    refs = [normalize(r["reference_text"]) for r in rows.values()]
    hyps = [normalize(r["asr"]) for r in rows.values()]
    out.setdefault("wer", {})[group] = jiwer.wer(refs, hyps)
Path("/kaggle/working/asr.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["wer"]), flush=True)
"""


def stage_job(kernel: str, output: Path, mode: str = "reference", dataset: str | None = None) -> Path:
    """Write a private CPU kernel (`convert.py` + metadata) without submitting it."""
    import json

    from kaggle_staging import stage_script

    if mode == "reference":
        if dataset is not None:
            raise ValueError("the reference job takes no dataset")
        bindings = {
            "KOKORO_VERSION": KOKORO_VERSION,
            "MISAKI_VERSION": MISAKI_VERSION,
            "REPO": REPO,
            "ASR_MODEL": ASR_MODEL,
            "UV_VERSION": UV_VERSION,
            "TORCH_CPU": TORCH_CPU,
            "ENGLISH_VOICES": ENGLISH_VOICES,
            "FIXTURES": FIXTURES,
            "GUTENBERG_URL": GUTENBERG_URL,
            "GUTENBERG_SENTENCES": GUTENBERG_SENTENCES,
            "INTERMEDIATE_ROWS": INTERMEDIATE_ROWS,
        }
        return stage_script(kernel, output, bindings, REFERENCE_JOB, (), gpu=False)
    if mode != "asr":
        raise ValueError("mode must be reference or asr")
    if dataset is None or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", dataset):
        raise ValueError("the asr job needs a dataset owner/slug")
    staged = stage_script(kernel, output, {"DATASET": dataset, "ASR_MODEL": ASR_MODEL},
                          ASR_JOB, (), gpu=False)
    path = staged / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset_sources"] = [dataset]
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="new private owner/slug")
    parser.add_argument("--output", type=Path, required=True, help="staging directory")
    parser.add_argument("--mode", choices=("reference", "asr"), default="reference")
    parser.add_argument("--dataset", help="private dataset owner/slug (asr mode)")
    args = parser.parse_args()
    print(stage_job(args.kernel, args.output, args.mode, args.dataset))


if __name__ == "__main__":
    main()
