# ComfyUI-MiniMax-H3-Infinity
Long-video windowing for **MiniMax H3** in ComfyUI. One node loops the H3 reference-to-video process (window → sample → re-anchor on the previous window's tail) and returns one continuous video+audio latent, so a single prompt can drive a minute-long clip, optionally locked to a master audio track.
