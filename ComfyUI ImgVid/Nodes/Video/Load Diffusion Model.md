
## **Purpose of the Node**

The **Load Diffusion Model** node loads the model weights so that they can be used by other nodes, like the **KSampler (Advanced)** node you showed earlier.  
Think of it as the foundation — without this, nothing else can generate or process images.

It outputs a **MODEL** object, which is then connected to samplers or other nodes that need the model to run.

---

## **Parameters**

There are two key settings here:

### **1. Model**

- This is where you select the actual `.safetensors` or `.ckpt` model file.
    
- In your screenshot, you’ve chosen:
    
    ```
    wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
    ```
    
    This suggests:
    
    - **wan2.2_t2v**: Likely a **WAN 2.2 text-to-video** model.
        
    - **high_noise**: Optimized for working with high-noise latents.
        
    - **14B**: Possibly the model size or parameter count.
        
    - **fp8_scaled**: Uses 8-bit floating point precision for reduced VRAM usage and potentially faster performance.
        

**What this affects:**

- The type of model determines what kind of tasks you can do:
    
    - **Text-to-image models (like Stable Diffusion 1.5/2.1)** – for generating still images.
        
    - **Text-to-video models** – for generating animations or videos.
        
    - **Custom models** – stylized generations, inpainting, etc.
        

---

### **2. weight_dtype (Weight Data Type)**

- Controls how the model’s weights are loaded into memory.
    
- Options typically include:
    
    - **default** – Automatically decides based on your system hardware and settings.
        
    - **fp32** – Full precision, highest VRAM use but maximum stability.
        
    - **fp16** – Half precision, much less VRAM usage, slightly faster, still high quality.
        
    - **bf16** – Another half-precision option, useful for some GPUs.
        
    - **fp8** – Ultra-low precision, significantly reduces VRAM usage but can slightly lower image fidelity.
        

**Why this matters:**

- If you’re running on a GPU with **limited VRAM**, using **fp16** or **fp8** helps you fit larger models and run more efficiently.
    
- If you want **maximum accuracy**, stick with **fp32** or **bf16** (if supported).
    

Since your model filename includes **fp8**, it's designed to run in **fp8 precision**, so setting `weight_dtype` to **default** is perfect — ComfyUI will automatically match the model’s format.

---

## **How It Connects to Other Nodes**

- **Output (MODEL)**:
    
    - This is the core object you’ll plug into nodes like **KSampler (Advanced)**.
        
    - Without this, the sampler won’t know which model to use.
        

Typical workflow:

```
Load Diffusion Model → KSampler (Advanced) → Decode Latent → Output
```

---

## **Performance Tips**

1. **Memory Management:**
    
    - If you have 8GB or less VRAM, go with **fp8** or **fp16**.
        
    - If you have 16GB+ VRAM, **bf16** or **fp32** is safe.
        
2. **Model Switching:**
    
    - If you want to switch between different models, you must replace the file here and reload the workflow.
        
3. **Using WAN 2.2:**
    
    - Since this is a text-to-video model, you’ll likely need:
        
        - A sampler that supports video.
            
        - Frame handling nodes after generation.
            

---

## **Summary Table**

|Setting|Description|When to Use|
|---|---|---|
|**Model**|Loads the `.safetensors` or `.ckpt` file|Always required|
|**weight_dtype**|Precision mode for loading model weights|`default` for auto, `fp16/fp8` for low VRAM, `fp32` for max quality|

