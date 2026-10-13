"""MiniMax H3 Infinity: a node that loops the MiniMax H3 reference-to-video
chunk -> sample -> re-anchor process internally to produce arbitrarily long
video *latent*, one prompt at a time, bounded by `total_seconds`. It only does
the windowing: model patches (LoRA, attention backend, sigma shift), the noise,
the sampler and the sigmas come in from regular nodes.

Modeled on nodes_scail_infinity.py's WanSCAILInfinity, adapted for the *real*
MiniMaxH3ReferenceToVideo API found in comfy_extras/nodes_minimax_h3.py:

  - Windows are specified in **seconds** (`window_seconds`, default 9.0)
    instead of a fixed frame count, converted to frames via `fps` (24, the
    model's native rate) and then snapped to the model's frame grid
    (frame_count % 17 == 5 -- see `align_frame_count` in nodes_minimax_h3.py).
  - Overlap is specified in **frames** (`overlap_frames`, default 5), rounded
    down to the clip grid 5, 22, 39... (17k + 5), the clip lengths
    MiniMaxH3AddGuide accepts. Chunks are 5k+2 latents and the anchor takes
    video_latent_t(overlap) of them, so the kept tail stays on the same
    FRAME_PER_TOKEN phase the previous chunk ended on.
  - `MiniMaxH3ReferenceToVideo` has no `model`, `negative`, or `previous_frames`-style
    re-anchor parameter of its own. Re-anchoring uses `MiniMaxH3AddGuide`: the
    previous chunk's tail is anchored at frame 0 of the next chunk (a keyframe on
    the target timeline, not a `ref_videos` entry, which only sits on a separate
    reference timeline and does not tie the new chunk to the old one).
  - `ref_images` / `ref_videos` / `ref_video_audios` / `ref_audios` are
    Autogrow inputs on the real node, i.e. plain dicts keyed
    "ref_image_0", "ref_image_1", ... -- not individual named parameters.

NO full-video VAE decode happens inside this node -- `latent` output is the
full AV NestedTensor pair (video + audio, both stitched, overlap-free),
same shape a real SamplerCustomAdvanced call would produce. Decode with
whatever VAE Decode / VAEDecodeAudio nodes you already trust (a tiled decode
is recommended for long videos) wired externally after this one.

The one place a *small* decode still happens internally is the re-anchor
step: `MiniMaxH3AddGuide` takes an IMAGE input, not a LATENT, so
each chunk's tail has to be decoded to pixels to hand to the next chunk.
`_decode_anchor` below decodes only the last couple of latent frames (just
enough to cover `overlap_frames` of pixels), not the whole chunk -- this is
unrelated to, and much cheaper than, decoding the full video for viewing.

It reuses ComfyUI / the MiniMax H3 node pack verbatim wherever possible:
  - comfy_extras.nodes_minimax_h3.MiniMaxH3ReferenceToVideo.execute() and
    MiniMaxH3AddGuide.execute() -> build the per-chunk conditioning + empty AV
    latent.
  - BasicGuider / SamplerCustomAdvanced from comfy_extras.nodes_custom_sampler
    -> sample each chunk with the `noise`, `sampler` and `sigmas` inputs.

Memory: peak VRAM equals a single window_seconds-long sampling run (no full
decode inflates it further). Every kept latent chunk is moved to CPU and flushed to disk immediately
(chunk_XXXX.pt under output/MiniMaxH3_Infinity_partial/<time>/). With free_between_windows (default on) all
models are also unloaded after each window.
After every chunk, GPU work is synchronized and both VRAM
(soft_empty_cache) and RAM (gc.collect) are cleared before the next chunk.
"""

from typing_extensions import override

import gc
import glob
import os
import time
import re

import torch
import comfy.nested_tensor

import nodes
import server
import folder_paths
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, io

from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide, align_frame_count,
    video_latent_t, AUDIO_LATENT_FPS, FPS as H3_NATIVE_FPS,
)
from comfy_extras.nodes_custom_sampler import (
    BasicGuider, SamplerCustomAdvanced,
)


def _run_stage(stage_name, window_index, total_windows, fn, *args, **kwargs):
    """Run one sub-step of a window (build conditioning / sample / anchor-decode) with
    a clear start/done/FAILED log line tagged by window+stage, so a crash can be
    pinned to an exact stage instead of just a bare core-code traceback."""
    tag = f"[window {window_index + 1}/{total_windows}][{stage_name}]"
    print(f"[MiniMaxH3-Infinity] {tag} starting", flush=True)
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        mem_info = ""
        # A GPU-level crash (e.g. a ROCm/hipErrorLaunchFailure "unspecified launch
        # failure") can leave the CUDA context itself unqueryable, so even this
        # read-only diagnostic call can raise. Never let that secondary failure
        # mask/replace the original exception below.
        if torch.cuda.is_available():
            try:
                mem_info = (f" | GPU memory at failure: allocated="
                            f"{torch.cuda.memory_allocated()/1e6:.1f}MB "
                            f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB")
            except Exception as mem_err:
                mem_info = (f" | (could not read GPU memory stats after the failure: "
                            f"{type(mem_err).__name__}: {mem_err} -- the CUDA/HIP context "
                            f"itself is likely dead, e.g. after an unspecified launch failure)")
        print(f"[MiniMaxH3-Infinity] {tag} !!! FAILED: {type(e).__name__}: {e}{mem_info}", flush=True)
        raise
    print(f"[MiniMaxH3-Infinity] {tag} done", flush=True)
    return result


def _check_finite(tensor, window_index, total_windows, stage):
    """Check a window's latent for NaN/Inf. Returns True if clean. Runs directly on
    the sampled latent -- no decode needed just to detect corruption."""
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        return True
    n_bad = (~finite_mask).sum().item()
    n_total = tensor.numel()
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    print(f"[MiniMaxH3-Infinity] !!!!! ALARM !!!!! window {window_index + 1}/{total_windows} "
          f"[{stage}] produced non-finite latent: {n_nan} NaN + {n_inf} Inf "
          f"out of {n_total} values ({100 * n_bad / n_total:.2f}%). "
          f"This window would decode to black/corrupted frames as-is.", flush=True)
    return False


def _video_tensor(samples):
    """Pull the video half out of a MiniMax H3 AV NestedTensor latent."""
    if hasattr(samples, "tensors"):
        return samples.tensors[0]
    return samples  # already a plain tensor (shouldn't normally happen for this model)


def _audio_tensor(samples):
    """Pull the audio half out of a MiniMax H3 AV NestedTensor latent, if present."""
    if hasattr(samples, "tensors") and len(samples.tensors) > 1:
        return samples.tensors[1]
    return None


def _sample_chunk(model, noise, sampler, sigmas, positive, latent, preview_prefix):
    guider = BasicGuider.execute(model=model, conditioning=positive).args[0]
    # read by Model Preview Override KJ to show the stitched video so far, not just this window
    guider.kj_preview_prefix = preview_prefix
    output, _denoised_output = SamplerCustomAdvanced.execute(
        noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent
    ).args
    return output


def _decode_anchor(vae, video_latent, overlap):
    """Decode just the tail of a video latent into the `overlap` pixel frames
    that anchor the next chunk. A chunk is 5k+2 latents and the tail
    video_latent_t(overlap) latents start on a 1-frame token (FRAME_PER_TOKEN
    phase 0), so that slice decodes to exactly `overlap` frames."""
    take = min(video_latent_t(overlap), video_latent.shape[2])
    imgs = vae.decode(video_latent[:, :, -take:])
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs[-overlap:].cpu()


def _save_latent_chunk(video_latent_cpu, audio_latent_cpu, out_dir, window_index):
    """Persist one chunk's kept (overlap-free) latent slice to disk immediately, so
    partial progress survives a crash in a later window. Saved as a genuine
    ComfyUI LATENT dict ({"samples": <NestedTensor or plain tensor>}), the same
    shape any other LATENT input expects -- load it back into a graph with the
    companion MiniMaxH3LoadInfinityChunk node below, no bespoke format to
    unpack by hand. audio_latent_cpu may be None if this chunk produced no
    audio, in which case the saved samples are a plain video tensor."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"chunk_{window_index:04d}.pt")
    samples = comfy.nested_tensor.NestedTensor((video_latent_cpu, audio_latent_cpu)) \
        if audio_latent_cpu is not None else video_latent_cpu
    torch.save({"samples": samples}, path)
    return path


def _audio_window(audio, start, seconds):
    """The `seconds` of a master track from `start`, zero-padded where the track ends early."""
    sr = audio["sample_rate"]
    wave = audio["waveform"][..., round(start * sr):round((start + seconds) * sr)]
    short = round(seconds * sr) - wave.shape[-1]
    if short > 0:
        wave = torch.nn.functional.pad(wave, (0, short))
    return {"waveform": wave, "sample_rate": sr}


def _load_finished_chunks(run_dir):
    """The (video, audio) latent slices of the windows a partial run already finished, in window order."""
    root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "MiniMaxH3_Infinity_partial"))
    run_dir = os.path.realpath(run_dir)
    if os.path.commonpath([run_dir, root]) != root:
        raise ValueError(f"resume_from must be a run folder under {root}")
    chunks = [torch.load(f, map_location="cpu", weights_only=False)["samples"]
              for f in sorted(glob.glob(os.path.join(run_dir, "chunk_*.pt")))]
    return [c.unbind() for c in chunks]


def _snap_window_frames(n, minimum=5):
    """Round n up to the model's frame grid: frame_count % 17 == 5."""
    return align_frame_count(max(minimum, n))


def _fit_window_frames(want, anchor):
    """The frame-grid window length closest to `want` frames (the anchor included)."""
    return max(_snap_window_frames(anchor + 1 if anchor else 5), 17 * round((want - 5) / 17) + 5)


_SHOT_HEADER_RE = re.compile(r'\[Shot\s+[^\]]*\]', re.IGNORECASE)
_BLOCK_SEPARATOR_RE = re.compile(r'^[ 	]*={3,}[ 	]*$', re.MULTILINE)
_DURATION_RE = re.compile(r'(?:duration|druation)\s*[:：=]\s*(\d+(?:\.\d+)?)\s*(?:s|sec|秒)?', re.IGNORECASE)


_SECTION_RE = re.compile(r'^(subject_definitions|summary|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music):', re.MULTILINE)
_NEGATIVE_RE = re.compile(r'^\[Negative:.*$', re.MULTILINE)
_SHOT_MARK_RE = re.compile(r'\[Shot\s*(\d+)\s*(?:[–-]\s*(\d+))?\]')
_TIMECODE_RE = re.compile(r'^\s*At\s+(\d+):(\d+(?:\.\d+)?)\s*[,，]?\s*')


def _split_by_shot(text):
    """[(first_shot, last_shot, text), ...] for a section written as `[Shot N] ... [Shot M] ...`."""
    marks = list(_SHOT_MARK_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((int(m.group(1)), int(m.group(2) or m.group(1)), text[m.end():end].strip()))
    return out


MIN_WINDOW_SECONDS = 5.0    # the model was trained on about 124-362 frames (5.2-15 s)
MAX_WINDOW_SECONDS = 14.5


def _timecode(seconds):
    return "At %02d:%06.3f" % (int(seconds // 60), seconds % 60)


def _shot_label(numbers):
    return "[Shot %d]" % numbers[0] if len(numbers) == 1 else "[Shot %d–%d]" % (numbers[0], numbers[-1])


def _group_shots(durations):
    """Indices of consecutive shots joined into windows of at least MIN_WINDOW_SECONDS (and at most
    MAX_WINDOW_SECONDS where possible); a short tail is merged into the window before it."""
    groups, current, total = [], [], 0.0
    for i, d in enumerate(durations):
        current.append(i)
        total += d
        if total >= MIN_WINDOW_SECONDS:
            groups.append(current)
            current, total = [], 0.0
    if current:
        if groups and sum(durations[i] for i in groups[-1] + current) <= MAX_WINDOW_SECONDS:
            groups[-1] += current
        else:
            groups.append(current)
    return groups


def _parse_official(prompt_text, fallback_seconds):
    """Split an official-format prompt (subject_definitions / summary / retention_analysis /
    detailed_description / overall_soundscape / non_diegetic_music / [Negative: ...]) into windows,
    as (duration_seconds, text) pairs. The shots are the [Shot N] of detailed_description only; a
    shot's length is the gap to the next shot's `At mm:ss.mmm` time code (the last one, or any shot
    without time codes, gets fallback_seconds). Consecutive shots are joined until a window is at
    least MIN_WINDOW_SECONDS, because the model was trained on clips of about 5 s and longer. Each
    window is a complete prompt of its own: shared sections kept, shot-specific ones cut down to its
    shots, renumbered from [Shot 1] with time codes counted from the window start. The summary keeps
    only its trailing sentence (the style line) -- its story part describes the whole video and would
    make every window act all of it out. Returns None if the prompt is not in this format."""
    marks = list(_SECTION_RE.finditer(prompt_text))
    sections = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(prompt_text)
        sections[m.group(1)] = prompt_text[m.end():end]
    if "detailed_description" not in sections:
        return None
    negative = ""
    for name, body in sections.items():
        n = _NEGATIVE_RE.search(body)
        if n:
            negative = n.group(0).strip()
            sections[name] = body[:n.start()] + body[n.end():]
    shots = _split_by_shot(sections["detailed_description"])
    if not shots:
        return None

    numbers, starts, bodies = [], [], []
    for i, (number, _, body) in enumerate(shots):
        tc = _TIMECODE_RE.match(body)
        starts.append(int(tc.group(1)) * 60 + float(tc.group(2)) if tc else (0.0 if i == 0 else None))
        numbers.append(number)
        bodies.append(body[tc.end():] if tc else body)
    durations = [starts[i + 1] - starts[i] if i + 1 < len(starts) and None not in (starts[i], starts[i + 1])
                 else fallback_seconds for i in range(len(starts))]

    soundscape = _split_by_shot(sections.get("overall_soundscape", ""))
    music = _split_by_shot(sections.get("non_diegetic_music", ""))
    summary = sections.get("summary", "").strip()
    if "。" in summary:
        summary = summary[summary.rindex("。") + 1:].strip()

    windows = []
    for group in _group_shots(durations):
        group_numbers = [numbers[i] for i in group]
        offsets, t = [], 0.0
        for i in group:
            offsets.append(t)
            t += durations[i]
        parts = []
        if "subject_definitions" in sections:
            parts.append("subject_definitions:\n" + sections["subject_definitions"].strip())
        if summary:
            parts.append("summary: [reference generation] " + summary)
        retention = []
        for line in sections.get("retention_analysis", "").strip().splitlines():
            appears = [int(n) for n in re.findall(r'\[Shot\s*(\d+)\]', line)]
            local = [k + 1 for k, n in enumerate(group_numbers) if n in appears]
            if local:
                retention.append(re.sub(r'\(appears in[^)]*\)',
                                        '(appears in ' + ', '.join('[Shot %d]' % k for k in local) + ')', line.strip()))
        if retention:
            parts.append("retention_analysis:\n" + "\n".join(retention))
        detailed = []
        for k, i in enumerate(group):
            prefix = "" if k == 0 else _timecode(offsets[k]) + ", "
            detailed.append("[Shot %d] %s%s" % (k + 1, prefix, bodies[i]))
        parts.append("detailed_description: " + " ".join(detailed))
        sound = ["[Shot %d] %s" % (k + 1, t) for a, b, t in soundscape for k, n in enumerate(group_numbers) if a <= n <= b]
        if sound:
            parts.append("overall_soundscape: " + " ".join(sound))
        mus = []
        for a, b, t in music:
            covered = [k + 1 for k, n in enumerate(group_numbers) if a <= n <= b]
            if covered:
                mus.append("%s %s" % (_shot_label(covered), t))
        if mus:
            parts.append("non_diegetic_music: " + " ".join(mus))
        if negative:
            parts.append(negative)
        windows.append((sum(durations[i] for i in group), "\n\n".join(parts)))
    return windows


def _parse_shots(prompt_text, fallback_seconds):
    """Split a storyboard-style prompt (`[Shot N]` headers, each optionally
    followed by a `Duration: X.X s` line) into a list of (duration_seconds_or_
    None, shot_text) pairs, one per shot -- each shot keeps only its own
    text, not the whole script. Any text before the first [Shot marker
    (character/scene descriptions, [ref_image_i] tag legends, etc.) is
    treated as a shared preamble and prepended to *every* shot's own text,
    since the model needs those tag definitions available in each call.
    A line of `===` splits the prompt into complete blocks instead, one per
    window, each used as written (no shared preamble) -- for the official
    subject_definitions / detailed_description format, whose own [Shot N]
    mentions would otherwise be cut up as headers.
    Returns None if no [Shot ...] header is found at all (not this format)."""
    blocks = _BLOCK_SEPARATOR_RE.split(prompt_text)
    if len(blocks) > 1:
        shots = []
        for block in blocks:
            block = block.strip()
            dur_match = _DURATION_RE.search(block)
            shots.append((float(dur_match.group(1)) if dur_match else None,
                          _DURATION_RE.sub('', block, count=1).strip() if dur_match else block))
        return shots
    official = _parse_official(prompt_text, fallback_seconds)
    if official is not None:
        return official
    matches = list(_SHOT_HEADER_RE.finditer(prompt_text))
    if not matches:
        return None

    preamble = prompt_text[:matches[0].start()].strip()
    shots = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt_text)
        body = prompt_text[start:end].strip()
        dur_match = _DURATION_RE.search(body)
        duration = float(dur_match.group(1)) if dur_match else None
        body_clean = _DURATION_RE.sub('', body, count=1).strip() if dur_match else body
        full_text = f"{preamble}\n\n{body_clean}".strip() if preamble else body_clean
        shots.append((duration, full_text))
    return shots


class MiniMaxH3Infinity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Infinity",
            display_name="MiniMax H3 Infinity (auto window)",
            category="video/minimax_h3",
            description=(
                "MiniMax H3 long-video windowing node. Internally loops "
                "window_seconds-long chunks (with an overlap_frames overlap, "
                "snapped to the model's 17k+5 frame grid) to stitch one "
                "continuous video *latent* from a single prompt. Outputs LATENT "
                "only -- decode it yourself with whichever VAE Decode node you "
                "trust. Default: 9s windows, 5-frame overlap."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Model.Input("model", tooltip="Fully patched model (LoRA, attention backend, sigma shift)."),
                io.Noise.Input("noise"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas", tooltip="Schedule used for every window."),
                io.Vae.Input("vae", tooltip="Needed both to encode reference images/video inside each "
                                             "MiniMaxH3ReferenceToVideo call, and for the small internal "
                                             "re-anchor decode between chunks (see module docstring). No full "
                                             "video is decoded through this node either way."),
                io.Vae.Input("audio_vae", optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                 tooltip="Either: (a) a single <Picture i> / <Video k> / <Audio j> style "
                                         "prompt for one continuous scene, applied to every window as-is; or "
                                         "(b) a storyboard script with [Shot N] headers -- turn on "
                                         "auto_split_shots (right below) to have each shot become its "
                                         "own window with its own text and duration, still anchored to the "
                                         "previous shot's tail for continuity."),
                io.Boolean.Input("auto_split_shots", default=False,
                                 tooltip="If the prompt contains [Shot N] headers (each optionally followed "
                                         "by a 'Duration: X.X s' line), split it: window count = shot count, "
                                         "each window gets only its own shot's text (any text before the "
                                         "first [Shot header is treated as a shared preamble and prepended "
                                         "to every shot), and each window's length comes from that shot's "
                                         "own Duration (falling back to window_seconds if a shot has none). "
                                         "The overlap/anchor mechanism still carries continuity across shot "
                                         "boundaries exactly as it does between windows in the non-split "
                                         "mode. window_seconds/total_seconds are ignored in this mode except "
                                         "as that fallback. An official-format prompt (subject_definitions / detailed_description "
                                         "with [Shot N] and `At mm:ss.mmm` time codes) is recognised: its shots are joined into "
                                         "windows of 5 s or more, each a complete prompt. A line of === instead splits the "
                                         "prompt into complete blocks, one per window, each used as written."),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Float.Input("window_seconds", default=9.0, min=0.5, max=120.0, step=0.5,
                               tooltip="Seconds of video per chunk at `fps`, snapped up to the model's "
                                       "17k+5 frame grid."),
                io.Int.Input("fps", default=24, min=1, max=120,
                             tooltip="MiniMax H3 is trained at 24 fps -- only change this if you know what "
                                     "you're doing."),
                io.Int.Input("overlap_frames", default=5, min=5, max=nodes.MAX_RESOLUTION,
                             tooltip="Frames of the previous chunk's tail anchored at frame 0 of the next "
                                     "chunk (MiniMaxH3AddGuide) and dropped from the stitched result. Rounded "
                                     "down to the clip grid 5, 22, 39... (17k + 5); 5 is the minimum."),
                io.Float.Input("total_seconds", default=0.0, min=0.0, max=3600.0, step=0.5,
                               tooltip="Target total output length in seconds. 0 = run a single window only "
                                       "(window_seconds long) and stop."),
                io.Boolean.Input("stop_on_corruption", default=True,
                                 tooltip="If a window comes out non-finite, stop the run here "
                                         "and return only the good windows so far, instead of anchoring every "
                                         "later window on corrupted data."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image, passed to every window"),
                        prefix="ref_image_", min=0, max=9)),
                io.String.Input("resume_from", default="", optional=True,
                                tooltip="A run folder under output/MiniMaxH3_Infinity_partial. Its finished chunks "
                                        "are kept as the first windows and generation continues with the next "
                                        "one (same prompt, settings and audio as the interrupted run), saving "
                                        "into that folder. Empty: start a new run."),
                io.Boolean.Input("free_between_windows", default=True, optional=True,
                                 tooltip="Unload all models and clean memory after every window, like POST "
                                         "/free {\"unload_models\": true}, so the next window starts from a clean "
                                         "state instead of slowing down."),
                io.Int.Input("windows_per_run", default=0, min=0, optional=True,
                             tooltip="Stop after this many new windows and return what is finished so far; 0 "
                                     "runs all windows. Queue the same prompt again with resume_from set to the "
                                     "run folder to continue. Every prompt starts from freshly loaded models, "
                                     "which is much faster than later windows of one long prompt."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Master track for the whole video. Each window anchors its own slice "
                                       "(including the overlap), so the soundtrack stays continuous and the "
                                       "performers sync to it. Needs audio_vae."),
                io.Image.Input("seed_ref_video", optional=True,
                               tooltip="Optional seed reference clip for window 1 only (fed in as ref_video_0). "
                                       "From window 2 onward the previous chunk's overlap tail is anchored "
                                       "at frame 0 instead."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="Concatenated, overlap-free AV latent "
                                  "(NestedTensor pair: video + audio, both stitched -- same shape a real "
                                  "SamplerCustomAdvanced output would have). Decode video with a video VAE "
                                  "Decode node and/or audio with VAEDecodeAudio, both wired externally."),
                io.Conditioning.Output(display_name="positive", tooltip="Positive conditioning from the final "
                                       "window's MiniMaxH3ReferenceToVideo call (exposed for inspection/"
                                       "chaining; each window rebuilds its own -- not a combined conditioning "
                                       "across windows)."),
                io.Int.Output(display_name="total_frames"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, model, vae, noise, sampler, sigmas, prompt, width, height,
                auto_split_shots, window_seconds, fps, overlap_frames, total_seconds,
                ref_image_size="match", audio_vae=None, stop_on_corruption=True,
                ref_images=None, seed_ref_video=None, audio=None, resume_from="", free_between_windows=True, windows_per_run=0):

        overlap = (max(5, overlap_frames) - 5) // 17 * 17 + 5
        if overlap != overlap_frames:
            print(f"[MiniMaxH3-Infinity] overlap_frames={overlap_frames} is not on the clip grid "
                  f"(5, 22, 39...) -- using {overlap}.", flush=True)

        # --- build the window plan: either one entry per parsed [Shot N], or the
        # existing uniform window_seconds/total_seconds chunking. Either way the
        # result is the same shape: a list of (this_length, anchor, window_prompt)
        # -- the shared loop below doesn't need to know which mode produced it. ---
        plan = []
        shots = _parse_shots(prompt, window_seconds) if auto_split_shots else None
        if auto_split_shots and shots is None:
            print("[MiniMaxH3-Infinity] auto_split_shots is on but no [Shot N] header was found in the "
                  "prompt -- falling back to normal single-scene mode (same full prompt on every window).",
                  flush=True)

        if shots is not None:
            print(f"[MiniMaxH3-Infinity] auto_split_shots: found {len(shots)} shot(s) in the prompt.", flush=True)
            planned = given = 0
            for i, (duration, shot_text) in enumerate(shots):
                anchor = 0 if i == 0 else overlap
                if duration is None:
                    print(f"[MiniMaxH3-Infinity]   shot {i + 1} has no 'Duration: X.X s' line -- falling "
                          f"back to window_seconds ({window_seconds}s) for its length.", flush=True)
                    duration = window_seconds
                # each window is fitted to the running total, so the grid rounding never piles up
                planned += round(duration * fps)
                this_length = _fit_window_frames(planned - given + anchor, anchor)
                given += this_length - anchor
                plan.append((this_length, anchor, shot_text))
                print(f"[MiniMaxH3-Infinity]   shot {i + 1}: requested {duration:.1f}s -> {this_length} "
                      f"frames (incl. {anchor}-frame anchor) | text: {shot_text[:80]!r}...", flush=True)
        else:
            window_length = _snap_window_frames(round(window_seconds * fps), minimum=overlap + 1)
            step = max(1, window_length - overlap)
            # total_seconds <= 0 -> single window_seconds-long run (back-compat default).
            # total_seconds > 0 -> exactly that length, even if it's *shorter* than window_seconds
            # (e.g. total_seconds=5 with window_seconds=9 gives one ~5s window, not a padded 9s one).
            target = window_length if total_seconds <= 0 else _snap_window_frames(round(total_seconds * fps), minimum=5)
            offset = 0
            i = 0
            while offset < target:
                anchor = 0 if i == 0 else overlap
                this_length = min(window_length, target - offset + anchor)
                this_length = _snap_window_frames(this_length, minimum=anchor + 1 if anchor else 5)
                plan.append((this_length, anchor, prompt))
                offset += this_length - anchor
                i += 1
            print(f"[MiniMaxH3-Infinity] target={target} frames (~{target / fps:.1f}s) | "
                  f"window={window_length} frames (~{window_length / fps:.1f}s, requested {window_seconds:.1f}s) "
                  f"overlap={overlap} step={step} -> {len(plan)} window(s)", flush=True)

        total_windows = len(plan)

        run_dir = resume_from.strip() or os.path.join(folder_paths.get_output_directory(),
                                                      "MiniMaxH3_Infinity_partial", time.strftime("%Y%m%d_%H%M%S"))
        print(f"[MiniMaxH3-Infinity] partial latent chunks will be saved to: {run_dir}", flush=True)

        stitched_video_lat = []
        stitched_audio_lat = []
        prev_anchor = None
        last_pos = None
        frames_done = 0
        audio_done = 0
        windows_done = 0

        if resume_from.strip():
            finished = _load_finished_chunks(run_dir)
            if not 0 < len(finished) < total_windows:
                raise ValueError(f"resume_from has {len(finished)} finished window(s), the plan has {total_windows}")
            stitched_video_lat = [v for v, _ in finished]
            stitched_audio_lat = [a for _, a in finished]
            windows_done = len(finished)
            frames_done = sum(FRAME_PER_TOKEN[k % 5] for k in range(sum(v.shape[2] for v in stitched_video_lat)))
            audio_done = sum(a.shape[3] for a in stitched_audio_lat)
            prev_anchor = _run_stage("anchor_decode", windows_done - 1, total_windows,
                                     lambda: _decode_anchor(vae, stitched_video_lat[-1], overlap))
            print(f"[MiniMaxH3-Infinity] resuming after {windows_done} finished window(s) "
                  f"({frames_done} frames, {frames_done / fps:.1f}s)", flush=True)

        for window_index, (this_length, anchor, window_prompt) in enumerate(plan):
            if window_index < windows_done:
                continue
            print(f"[MiniMaxH3-Infinity] window {window_index + 1}/{total_windows}: "
                  f"requesting {this_length} frames | anchored on {anchor} previous frame(s)", flush=True)

            ref_videos = {"ref_video_0": seed_ref_video} if window_index == 0 and seed_ref_video is not None else None

            def build_conditioning():
                pos, latent = MiniMaxH3ReferenceToVideo.execute(
                    clip=clip, vae=vae, audio_vae=audio_vae, prompt=window_prompt,
                    width=width, height=height, length=this_length,
                    ref_image_size=ref_image_size,
                    ref_images=ref_images, ref_videos=ref_videos,
                    ref_video_audios=None, ref_audios=None,
                ).args
                if prev_anchor is not None:
                    pos = MiniMaxH3AddGuide.execute(positive=pos, latent=latent, frame_idx=0,
                                                    vae=vae, image=prev_anchor).args[0]
                if audio is not None:
                    pos = MiniMaxH3AddGuide.execute(positive=pos, latent=latent, frame_idx=0, audio_vae=audio_vae,
                                                    audio=_audio_window(audio, (frames_done - anchor) / fps, this_length / fps)).args[0]
                return pos, latent

            pos, latent = _run_stage("build_conditioning", window_index, total_windows, build_conditioning)

            # What this window keeps: the anchor is a 17k+5 clip, which occupies video_latent_t(anchor)
            # leading latents; chunks are 5k+2 latents, so the kept tail starts on the same
            # FRAME_PER_TOKEN phase the previous chunk ended on. Audio keeps whatever makes the stitched
            # audio track as long as the stitched video, so rounding never accumulates across joins.
            lat_drop_video = video_latent_t(anchor) if anchor else 0
            frames_done += this_length - anchor
            n_keep_audio = round(frames_done / H3_NATIVE_FPS * AUDIO_LATENT_FPS) - audio_done
            lat_drop_audio = round(this_length / H3_NATIVE_FPS * AUDIO_LATENT_FPS) - n_keep_audio

            preview_prefix = None
            if stitched_video_lat:
                preview_prefix = {
                    "video": torch.cat(stitched_video_lat, dim=2),
                    "audio": torch.cat(stitched_audio_lat, dim=3) if stitched_audio_lat else None,
                    "video_drop": lat_drop_video,
                    "audio_drop": lat_drop_audio,
                }

            sampled = _run_stage(
                "sampling", window_index, total_windows,
                lambda: _sample_chunk(model, noise, sampler, sigmas, pos, latent, preview_prefix)
            )
            chunk_samples = sampled["samples"]
            video_latent = _video_tensor(chunk_samples)
            audio_latent = _audio_tensor(chunk_samples)

            clean = _check_finite(video_latent, window_index, total_windows, "sampling (video)")
            if audio_latent is not None:
                clean = _check_finite(audio_latent, window_index, total_windows, "sampling (audio)") and clean

            if not clean:
                print(f"[MiniMaxH3-Infinity] !!!!! ALARM !!!!! window {window_index + 1}/{total_windows} "
                      f"is non-finite.", flush=True)
                if stop_on_corruption:
                    total_so_far = sum(t.shape[2] for t in stitched_video_lat)
                    print(f"[MiniMaxH3-Infinity] Stopping here to avoid anchoring later windows on corrupted "
                          f"data. Returning the {total_so_far} good latent frame(s) generated before this "
                          f"window.", flush=True)
                    del sampled, latent, chunk_samples, video_latent, audio_latent, pos
                    gc.collect()
                    mm.soft_empty_cache()
                    break
                print("[MiniMaxH3-Infinity] stop_on_corruption is off -- continuing with the corrupted "
                      "window anyway (expect black/garbled frames from here on).", flush=True)

            # --- stitch: drop the anchored prefix on every chunk after the first ---
            kept_video = video_latent[:, :, lat_drop_video:]
            kept_audio = None
            if audio_latent is not None:
                kept_audio = audio_latent[:, :, :, lat_drop_audio:]
                audio_done += n_keep_audio
            stitched_video_lat.append(kept_video.cpu())
            if kept_audio is not None:
                stitched_audio_lat.append(kept_audio.cpu())
            total_so_far = sum(t.shape[2] for t in stitched_video_lat)
            print(f"[MiniMaxH3-Infinity]   sampled {video_latent.shape[2]} video latent frames, kept "
                  f"{kept_video.shape[2]} new (lat_drop_video={lat_drop_video}) "
                  f"(stitched total: {total_so_far} latent frames)", flush=True)

            saved_path = _save_latent_chunk(kept_video.cpu(), kept_audio.cpu() if kept_audio is not None else None,
                                             run_dir, window_index)
            print(f"[MiniMaxH3-Infinity]   saved chunk to {saved_path}", flush=True)

            # --- next anchor: decode just the tail into pixels for the next chunk's AddGuide ---
            if window_index + 1 < total_windows:
                prev_anchor = _run_stage(
                    "anchor_decode", window_index, total_windows,
                    lambda: _decode_anchor(vae, video_latent, overlap)
                )
            last_pos = pos
            del sampled, latent, chunk_samples, video_latent, audio_latent, kept_video, kept_audio, pos
            if free_between_windows:
                mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()

            window_index += 1
            if windows_per_run and window_index - windows_done >= windows_per_run:
                break

        if not stitched_video_lat:
            raise RuntimeError("MiniMax H3 Infinity: the first window produced a non-finite latent, nothing to return.")
        result_video_lat = torch.cat(stitched_video_lat, dim=2)
        total_latent_frames = result_video_lat.shape[2]
        total_pixel_frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(total_latent_frames))

        if stitched_audio_lat:
            result_audio_lat = torch.cat(stitched_audio_lat, dim=3)
            result_samples = comfy.nested_tensor.NestedTensor((result_video_lat, result_audio_lat))
            print(f"[MiniMaxH3-Infinity] done: {total_latent_frames} video latent frames + "
                  f"{result_audio_lat.shape[3]} audio latent frames over {window_index} window(s). "
                  f"Decode externally (video VAE Decode + VAEDecodeAudio) to get pixels/waveform.", flush=True)
        else:
            result_samples = result_video_lat
            print(f"[MiniMaxH3-Infinity] done: {total_latent_frames} video latent frames over "
                  f"{window_index} window(s). No audio latent was produced -- decode externally to get "
                  f"pixels.", flush=True)

        return io.NodeOutput({"samples": result_samples}, last_pos, total_pixel_frames)


class MiniMaxH3LoadInfinityChunk(io.ComfyNode):
    """Loads a single chunk .pt file saved by MiniMaxH3Infinity's crash-recovery
    path back into the graph as a normal LATENT. Point this at any
    chunk_XXXX.pt under MiniMaxH3_Infinity_partial/<run>/ to inspect or decode
    partial progress from an interrupted run, without needing to re-run the
    whole thing. To manually stitch several chunks back into one longer clip,
    load each with a separate copy of this node and use a Latent Batch /
    concat-style node downstream (concatenating along the video's temporal
    axis, dim=2, and the audio's, dim=3, if present)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoadInfinityChunk",
            display_name="Load MiniMax H3 Infinity Chunk",
            category="video/minimax_h3",
            description="Loads one chunk_XXXX.pt file (saved by MiniMax H3 Infinity's crash-recovery "
                        "path), or all chunks of a run folder joined in order, back in as a normal LATENT.",
            inputs=[
                io.String.Input("path", tooltip="Full path to a chunk_XXXX.pt file, or to a run folder, under "
                                                 "this install's output/MiniMaxH3_Infinity_partial/."),
                io.Boolean.Input("auto_merge", default=True, optional=True,
                                 tooltip="If path is a run folder, join all its chunk_*.pt in order (video along "
                                         "the latent time axis, audio along its last axis) before the decode "
                                         "starts. Off: path must be a single chunk file."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, path, auto_merge=True) -> io.NodeOutput:
        root = os.path.realpath(os.path.join(folder_paths.get_output_directory(), "MiniMaxH3_Infinity_partial"))
        path = os.path.realpath(path)
        if os.path.commonpath([path, root]) != root:
            raise ValueError(f"chunk path must be under {root}")
        if os.path.isdir(path):
            if not auto_merge:
                raise ValueError("path is a folder: turn on auto_merge or point to one chunk_XXXX.pt file")
            chunks = [torch.load(f, map_location="cpu", weights_only=False)["samples"]
                      for f in sorted(glob.glob(os.path.join(path, "chunk_*.pt")))]
            video, audio = zip(*(c.unbind() for c in chunks))
            samples = comfy.nested_tensor.NestedTensor((torch.cat(video, dim=2), torch.cat(audio, dim=3)))
            return io.NodeOutput({"samples": samples})
        data = torch.load(path, map_location="cpu", weights_only=False)
        return io.NodeOutput(data)


class UnloadModelsAfterRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UnloadModelsAfterRun",
            display_name="Unload Models After Run",
            category="utils/memory",
            description="Same as POST /free {\"unload_models\": true}: once this prompt finishes, all models "
                        "are unloaded and memory is cleaned. Cached node outputs are kept.",
            inputs=[],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls):
        return float("nan")

    @classmethod
    def execute(cls) -> io.NodeOutput:
        server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
        return io.NodeOutput()


class FreeMemoryAfterRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FreeMemoryAfterRun",
            display_name="Free Memory After Run",
            category="utils/memory",
            description="Same as POST /free {\"free_memory\": true}: once this prompt finishes, all models "
                        "are unloaded, memory is cleaned and the node output cache is reset, so the next "
                        "prompt runs every node again.",
            inputs=[],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls):
        return float("nan")

    @classmethod
    def execute(cls) -> io.NodeOutput:
        server.PromptServer.instance.prompt_queue.set_flag("free_memory", True)
        return io.NodeOutput()


class MiniMaxH3InfinityExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [MiniMaxH3Infinity, MiniMaxH3LoadInfinityChunk, UnloadModelsAfterRun, FreeMemoryAfterRun]


async def comfy_entrypoint():
    return MiniMaxH3InfinityExtension()
