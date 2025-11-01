
Create Container
   ```bash
     jetson-containers run --name -v $(pwd):/workspace $(autotag ollama) bash
     ```
---
Make a Modelfile
   ```bash
	FROM tinyllama:1.1b
	PARAMETER num_ctx 256
	PARAMETER num_gpu 10
     ```
---
create a new LLM
   ```bash
     ollama create <model-name> -f /workspace/Modelfile
     ```
---
Run Code
   ```bash
     echo "hello" | ollama run tinyllama-jetson
     ```
---




   ```bash
     OLLAMA_LLM_LIBRARY=cuda_v12 ollama serve > /tmp/ollama.log 2>&1 &
     ```

   ```bash
     OLLAMA_LLM_LIBRARY=cuda_v12 ollama serve > /tmp/ollama.log 2>&1 &
     ```

   ```bash
     OLLAMA_LLM_LIBRARY=cuda_v12 ollama serve > /tmp/ollama.log 2>&1 &
     ```
