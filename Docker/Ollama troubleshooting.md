## Ollama Troubleshooting on Jetson

### Summary of the Fix
- Limited TinyLlama’s GPU footprint by setting `num_gpu 10` in the shared `Modelfile`, keeping context at 256 tokens.
- Created two local variants inside the Jetson containers image:
  - `tinyllama-jetson` – offloads 10 layers to CUDA (fits in ~2.5 GiB VRAM).
  - `tinyllama-cpu` – forces CPU backend for a guaranteed fallback.
- Restarted the Ollama server with the desired backend (`cuda_v12` or `cpu`) before running prompts.

### Current Files & Locations
- Host/shared Modelfile: `/home/jetson/Projects/project-containers/Modelfile`

```3:3:/home/jetson/Projects/project-containers/Modelfile
FROM tinyllama:1.1b
PARAMETER num_ctx 256
PARAMETER num_gpu 10
```

### Workflow / How To Use It
1. **Enter the container** (already running as `test1`, otherwise use `jetson-containers run $(autotag ollama)`).
2. **Pick a backend**:
   - GPU (preferred):
     ```bash
     OLLAMA_LLM_LIBRARY=cuda_v12 ollama serve > /tmp/ollama.log 2>&1 &
     ```
   - CPU fallback:
     ```bash
     OLLAMA_LLM_LIBRARY=cpu ollama serve > /tmp/ollama.log 2>&1 &
     ```
   - Confirm the server is ready with `ollama list`.
3. **Run a model**:
   ```bash
   echo "hello" | ollama run tinyllama-jetson   # GPU path
   echo "hello" | ollama run tinyllama-cpu      # CPU fallback
   ```
4. **Watch logs** if something stalls:
   ```bash
   tail -f /tmp/ollama.log
   ```
5. **Shut down** by killing the serve process when done:
   ```bash
   pkill ollama
   ```

### Future Tweaks / Ideas
- If CUDA OOM reappears, lower `num_gpu` (e.g. 8) or reduce `num_ctx`.
- Free additional GPU memory by stopping other CUDA workloads (`sudo nvpmodel`, `sudo systemctl stop jetson-io` etc.).
- Tune `OLLAMA_GPU_OVERHEAD` and `OLLAMA_MAX_LOADED_MODELS` if juggling multiple models.
- Automate backend selection inside `start_ollama` so the container boot script remembers your preference.
- Add a small wrapper script to restart the server with the desired backend and model (e.g. `run_tinyllama_gpu.sh`).

### Quick Reference Commands
- List cached models: `ollama list`
- Show model metadata: `ollama show tinyllama-jetson`
- Recreate the GPU-limited model if the Modelfile changes:
  ```bash
  ollama create tinyllama-jetson -f /workspace/Modelfile
  ```
- Unload a model without stopping the server:
  ```bash
  ollama unload tinyllama-jetson
  ```
- Stop the Ollama server (frees memory and logs out):
  ```bash
  pkill ollama
  ```

### Troubleshooting Checklist
- `Error: could not connect to ollama app` → ensure `ollama serve` is running.
- `unable to allocate CUDA0 buffer` → decrease `num_gpu`, reduce context, or switch to CPU backend.
- `cudaMalloc failed: out of memory` → verify no other processes are heavy on VRAM (`tegrastats`, `nvidia-smi` for dGPU systems).
- `unknown parameter` when creating models → confirm syntax matches supported Modelfile parameters (`num_ctx`, `num_gpu`, etc.).

### System RAM Headroom (CPU vs CUDA)
- Current container snapshot (`free -h`) shows ~`2.5 GiB` of available system RAM with either backend active (total 7.4 GiB, swap 3.7 GiB).
- **TinyLlama Q4_0 (~1.7 GiB weights + ~0.2 GiB KV for 256 ctx)** leaves only ~`0.6 GiB` of headroom before the kernel starts dipping into cache/swap.
- **CUDA (`cuda_v12`) backend** still maps the weights in RAM, but offloads compute to the GPU. Realistically anything bigger than ~2 GiB GGUF will push you into swapping or OOM, so stick to ≤2–3 B models in heavy quantizations if you experiment.
- **CPU-only backend** keeps everything in system memory; it can fall back to swap, but expect a major slowdown. Treat ~2 GiB GGUF as the practical ceiling unless you expand RAM or close other services.
- Remember the KV cache scales linearly with `num_ctx`, so doubling context to 512 roughly doubles the extra 0.2 GiB footprint.

Document created: 2025-10-31

