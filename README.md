# ComfyUI-MiniMax-H3-Infinity

Long-video windowing for **MiniMax H3** in ComfyUI. One node loops the H3 reference-to-video process
(window → sample → re-anchor on the previous window's tail) and returns one continuous video+audio latent,
so a single prompt can drive a minute-long clip, optionally locked to a master audio track.

> Experimental. Written and tested on one machine (Windows 11, RX 9060 XT 16 GB, 40 GB RAM, ROCm build of ComfyUI 0.39).
> It reuses ComfyUI's own `MiniMaxH3ReferenceToVideo` / `MiniMaxH3AddGuide` / `SamplerCustomAdvanced`, so it needs a
> ComfyUI version with native MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`).

## Demo

[![MiniMax H3 Infinity demo](assets/mv_preview.gif)](https://youtube.com/shorts/V5FETbVz8fk)

The example music video made with this node (113 s, 13 windows, MiniMax H3 + MiniMax Music 3). Click the preview to watch the full video on [YouTube](https://youtube.com/shorts/V5FETbVz8fk).

## Nodes

| Node | What it does |
|---|---|
| **MiniMax H3 Infinity (auto window)** | Plans windows, samples each one, anchors it on the previous tail, stitches the latents, saves every finished window to `output/MiniMaxH3_Infinity_partial/<time>/chunk_XXXX.pt`. Outputs `latent` (video + audio NestedTensor), `positive`, `total_frames`. |
| **Load MiniMax H3 Infinity Chunk** | Loads one `chunk_XXXX.pt`, or a whole run folder joined in order (`auto_merge`), as a normal LATENT for decoding. |
| **Unload Models After Run** | Same as `POST /free {"unload_models": true}` once the current prompt finishes. |
| **Free Memory After Run** | Same as `POST /free {"free_memory": true}` (also resets the node cache). |

No full-video decode happens inside the node. Wire `LTXVSeparateAVLatent` → a tiled VAE decode (`VAEDecodeTiled`,
tile 1024 / overlap 64 / temporal 64 / 8 worked well) → `CreateVideo`.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/blackmeat1225/ComfyUI-MiniMax-H3-Infinity
```

Restart ComfyUI. Model files are the usual MiniMax H3 ones
([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)): the ref2va DiT, the text encoder, the video and audio VAEs,
and optionally a Turbo LoRA.

## Quick start

1. The example is the music video this node was built for: a 113 s Christmas MV, 13 windows, one singer from a reference
   image, locked to a song (generated with MiniMax Music 3). Load `examples/mv_workflow_api.json` (API format, e.g. by queueing
   it over HTTP or with a *Load (API)* extension). Put your own reference image and song in `ComfyUI/input/`, then change
   `singer.png` / `mv_song.wav` and the model file names to yours.
2. The prompt is in the official MiniMax H3 format, see `examples/mv_prompt.txt` (13 shots with time codes; each sung shot
   carries its lyrics in `<d>…</d>`, which only strengthens the text constraint and does not guarantee phoneme alignment). With `auto_split_shots` on, the
   `[Shot N]` entries and their `At mm:ss.mmm` time codes become windows (consecutive shots are merged until a window is
   at least 5 s; a window is at most ~14.5 s). Each window gets its own complete prompt.
3. Decode the result with the nodes after the Infinity node.

### Main inputs

| Input | Meaning |
|---|---|
| `prompt`, `auto_split_shots` | One scene prompt, or a storyboard split into windows (official format, or `[Shot N]` / `===` blocks). |
| `window_seconds`, `fps`, `overlap_frames` | Window length (fallback when a shot has no time code), frame rate (24), anchor length (5, 22, 39… = 17k+5). |
| `total_seconds` | Total length without shots; 0 = one window. With shots the length comes from the time codes. |
| `ref_images` | Reference images, passed to every window. |
| `audio` + `audio_vae` | Master track. Each window anchors its own slice, so the soundtrack stays continuous and the singer follows it. |
| `resume_from` | A run folder under `output/MiniMaxH3_Infinity_partial/`. Finished chunks are kept and generation continues after them. |
| `windows_per_run` | Stop after N new windows (0 = all). Queue the prompt again with `resume_from` set to continue. |
| `free_between_windows` | Unload all models and clean memory after every window (on by default). |
| `stop_on_corruption` | Stop if a window comes out non-finite instead of anchoring later windows on it. |

Window lengths follow the model's frame grid (frames = 17k + 5) and are fitted cumulatively, so the total length does not
drift (nine 10 s shots come out at 90.17 s).

The example uses only ComfyUI core nodes plus this package. In my own run I also added the third-party `H3SLAAttentionROCM`
and `TESpeedMiniMaxH3` nodes after the attention backend; they are not needed.

## Long runs, speed and memory (what I measured)

* A prompt that starts from freshly loaded models was fastest: on the setup above a 10 s window at 0.5 MP took ~10 minutes
  in the first prompt after a restart, while later windows of the same long prompt were up to ~3× slower. Running one
  window per prompt (`windows_per_run = 1`, `Free Memory After Run` at the end) kept every window fast.
  `tools/run_per_window.py` does the queueing for you:
  ```bash
  python tools/run_per_window.py examples/mv_workflow_api.json --windows 13 --output-dir /path/to/ComfyUI/output
  ```
* Keep windows at about 9–10 s or less; very long sequences got slow or unstable here.
* Decode with a tiled VAE decode; the plain VAE decode of a 90 s latent took hours.
* Every window is saved as it finishes, so an interrupted run can be continued with `resume_from` (same prompt, settings and audio).

## Limitations

* Lip-sync with a master track follows the audio guide but is not exact; in my tests the mouth could lead the audio by a fraction of a second to about a second.
* The outfit can drift between windows when a shot changes framing, and the model sometimes cuts to a new shot inside a window.
* Tested with one hardware/software combination only.

## 繁體中文簡介

範例 MV：[YouTube Shorts](https://youtube.com/shorts/V5FETbVz8fk)

MiniMax H3 的長影片節點：把一段分鏡腳本（官方格式，`[Shot N]` 加 `At mm:ss.mmm`）切成多個視窗，每個視窗用上一個視窗的尾端當錨點，
串成一段連續的影像＋聲音 latent，可選用一整首歌當主音軌讓人物對嘴。每個視窗完成就存檔，中斷後可用 `resume_from` 接續；
`windows_per_run` 搭配 `Free Memory After Run` 可以讓每個視窗都從乾淨的狀態開始（我的環境下快很多）。
安裝：把資料夾放進 `ComfyUI/custom_nodes/`，重啟，載入 `examples/mv_workflow_api.json`（就是我做 MV 用的範例）。

## License

GPL-3.0 (see `LICENSE`). The windowing approach follows ComfyUI's own `WanSCAILInfinity` node and the node calls into ComfyUI's MiniMax H3 nodes.
