
### **Step 1: Create Container**

```bash
jetson-containers run --name <name> -v $(pwd):/workspace $(autotag ollama) bash
```

---

### **Step 2: Make a Modelfile**

Create a `Modelfile` in your workspace:

```bash
FROM tinyllama:1.1b
PARAMETER num_ctx 256
PARAMETER num_gpu 10
```

---

### **Step 3: Create a New LLM**

```bash
ollama create <model-name> -f /workspace/Modelfile
```

---

### **Step 4: Set CUDA Version and Start Server**

```bash
OLLAMA_LLM_LIBRARY=cuda_v12 ollama serve > /tmp/ollama.log 2>&1 &
```

---

### **Step 5: Enter LLM Conversation**

```bash
ollama run tinyllama-jetson
```

---

### **Step 6: Run Code**

```bash
echo "hello" | ollama run tinyllama-jetson
```

---

### **Step 7: Update an Existing Model**

If you need to modify an existing model, delete and recreate it:

```bash
ollama rm <model>  # Delete old phi model
ollama create <model> -f /workspace/<modelfile>
```

---

### **Step 8: Stop a Running Model**

```bash
ollama stop <model>
```

---
