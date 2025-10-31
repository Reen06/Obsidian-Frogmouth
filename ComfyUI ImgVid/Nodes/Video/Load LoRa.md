

## **Purpose of the LoraLoaderModelOnly Node**

- Loads a **LoRA file** and merges it with a **base diffusion model** dynamically at runtime.
    
- Instead of baking changes permanently into the model, this node temporarily applies the LoRA for the current generation.
    
- Useful for:
    
    - Adding **specific characters** or styles.
        
    - Making **minor adjustments** without retraining the entire model.
        
    - Working with **text-to-video workflows**, like WAN 2.2 T2V, for specialized motion or behavior.
        

---

## **Parameters**

### **1. model**

- The LoRA file to be loaded (`.safetensors` or `.pt`).
    
- In your screenshot, the selected file is:
    
    ```
    wan2.2_t2v_lightx2v_4steps_lora_...
    ```
    

**What this name suggests:**

- `wan2.2_t2v` → Designed for **WAN 2.2 Text-to-Video**.
    
- `lightx2v` → Possibly a specific motion style or refinement type for video generation.
    
- `4steps` → Likely optimized for a 4-step merge process, meaning it's lightweight and fast.
    

**Important Note:**  
The LoRA must **match the base diffusion model type**.

- Since you’re using **WAN 2.2**, you must use WAN 2.2-compatible LoRAs.
    
- Mixing mismatched LoRAs can lead to:
    
    - Artifacts in frames.
        
    - Flickering in generated videos.
        
    - Complete generation failure.
        

---

### **2. strength_model**

- A floating-point value that controls how strongly the LoRA influences the base model.
    

|Value|Effect Description|
|---|---|
|**0.0**|LoRA has no effect (disabled).|
|**0.3–0.5**|Subtle influence (mild effect).|
|**0.7–1.0**|Strong influence (default is **1.0**).|
|**>1.0**|Overpowering effect, may cause artifacts.|

**Recommendation:**

- Start at **1.0**, then adjust lower if the LoRA is too aggressive.
    
- For video workflows, **0.7–0.9** is often better to avoid frame inconsistencies.
    

---

## **Output: MODEL**

- This node outputs a **merged model object**:
    
    - Combines your **base model** (e.g., WAN 2.2 T2V) with the LoRA’s modifications.
        
    - Feeds directly into the **KSampler (Advanced)** node or other model-dependent nodes.
        

Typical pipeline connection:

```
Load Diffusion Model → LoraLoaderModelOnly → KSampler (Advanced)
```

---

## **How It Fits Into Your WAN 2.2 T2V Video Workflow**

Here’s how it fits with all your nodes so far:

1. **Load Diffusion Model** → Load the WAN 2.2 base model.
    
2. **LoraLoaderModelOnly** → Merge the WAN-compatible LoRA with the base model.
    
3. **Load CLIP** → Interpret your text prompt.
    
4. **EmptyHunyuanLatentVideo** → Create a blank latent video canvas.
    
5. **KSampler (Advanced)** → Generate video frames using the merged model + prompt.
    
6. **Load VAE + VAE Decode** → Convert final latents into actual video frames.
    

Pipeline structure:

```
Load Diffusion Model
      ↓
LoraLoaderModelOnly
      ↓
KSampler (Advanced)
```

---

## **Why Use This Instead of a Regular Model Loader**

- LoRAs are much **smaller** than full models (tens or hundreds of MB vs. several GB).
    
- You can **switch styles or behaviors instantly** without loading a completely new model.
    
- Perfect for video workflows where experimenting with different styles is common.
    

---

## **Performance Tips**

1. **Keep VRAM in Check:**
    
    - LoRAs are lightweight, but each active LoRA adds a bit of extra VRAM usage.
        
    - Multiple LoRAs stacked can quickly consume more resources.
        
2. **Start with Default Strength:**
    
    - Start at **1.0**, then lower if results look unnatural or over-stylized.
        
3. **One LoRA at a Time:**
    
    - Especially for video generation, stick to one LoRA until you confirm smooth, stable frames.
        
    - Combining multiple LoRAs can cause flickering between frames.
        
4. **Check for Model Compatibility:**
    
    - Always confirm that the LoRA was trained for **WAN 2.2 T2V** to avoid output issues.
        

---

## **Comparison: LoraLoaderModelOnly vs. Other LoRA Nodes**

|Node Type|Purpose|
|---|---|
|**LoraLoaderModelOnly**|Only merges the LoRA with the model — light and clean.|
|**LoraLoaderFull**|Loads LoRA and related prompt encoders — heavier but more flexible.|

Since you're already using a **Load CLIP** node and a **Load Diffusion Model** node, `LoraLoaderModelOnly` is perfect here.

---

## **Summary Table**

|Parameter|Description|Best Practice|
|---|---|---|
|**model**|LoRA file to merge with base model|Use WAN 2.2 T2V-specific LoRAs|
|**strength_model**|How strongly the LoRA affects generation|Start at 1.0, adjust between 0.7–1.0 for video|

---

## **Example Use Case**

- Base model: `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
    
- LoRA: `wan2.2_t2v_lightx2v_4steps_lora.safetensors`
    
- Goal: Create a short 640x640 81-frame video with a subtle stylization effect.
    
- Recommended `strength_model`: **0.85**
    

