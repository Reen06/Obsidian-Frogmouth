
## **Purpose of EmptyHunyuanLatentVideo**

- Generates an **empty latent video** (a block of uninitialized latent frames) that serves as the starting point for the video generation pipeline.
    
- Similar to starting with random noise for a single image, but here it’s **multi-frame noise** for video.
    
- Provides:
    
    - Video **resolution** (width & height).
        
    - Video **length** (number of frames).
        
    - Batch configuration for generating multiple video clips in one run.
        

This node is **essential for WAN 2.2 or similar video models**, since those models need a latent video input to operate.

---

## **Parameters and Their Functions**

### **1. width**

- The **width in pixels** of the video frames in the latent space.
    
- This determines the horizontal resolution of the output video.
    

**Typical values:**

- 512, 640, 768
    
- Must usually be divisible by **64** (common for diffusion-based models).
    

**Example:**

- Setting width to `640` → final video will have a width of 640 pixels.
    

---

### **2. height**

- The **height in pixels** of the video frames in the latent space.
    
- Like width, this must usually be a multiple of **64**.
    

**Example:**

- Height of `640` → each frame will be 640 pixels tall.
    

---

### **3. length**

- The **number of frames** in the video.
    
- Higher numbers = longer videos, but also more VRAM and generation time.
    

**Guidelines:**

|Frames (`length`)|Approx Video Duration (at 24 fps)|
|---|---|
|24|1 second|
|48|2 seconds|
|81|~3.4 seconds|
|96|4 seconds|

So, your current value of **81 frames** ≈ 3.4 seconds of video at 24 FPS.

---

### **4. batch_size**

- How many separate video sequences to generate in parallel.
    
- **1** = just a single video.
    
- **>1** = generate multiple videos at once (requires much more VRAM).
    

**Example:**

- `batch_size = 2` → The node will output two latent videos, each independent.
    

---

## **Output: LATENT**

- Produces an **empty latent video tensor** that other nodes can work with.
    
- Typically connected to:
    
    - **KSampler (Advanced)** → to add noise and denoise based on your prompt.
        
    - Then to **VAE Decode** → to turn the latent frames into actual video frames.
        

---

## **Typical Workflow Integration**

Here’s where this node fits in a **text-to-video pipeline** with your WAN model:

```
Load CLIP → CLIP Text Encode (Positive / Negative Prompts)

EmptyHunyuanLatentVideo → KSampler (Advanced) → VAE Decode → Video Output
```

1. **EmptyHunyuanLatentVideo** – creates the blank latent video space.
    
2. **KSampler (Advanced)** – fills in the latent with generated frames based on prompts and model guidance.
    
3. **VAE Decode** – converts the completed latent frames to pixel space for viewing/export.
    

---

## **Example Settings for Common Use Cases**

|Goal|Width x Height|Length (Frames)|Notes|
|---|---|---|---|
|Fast preview|512 x 512|24|Quick test, very low VRAM.|
|Short clip|640 x 640|48|2-second video.|
|Medium length|640 x 640|81|~3.4-second video.|
|High-quality clip|768 x 768|96|Requires high VRAM (16GB+).|

---

## **Performance Tips**

1. **VRAM Usage Factors:**
    
    - **Width × Height × Length × Batch Size**  
        The larger these numbers, the more VRAM needed.
        
    - Example: Doubling width or height quadruples VRAM usage.
        
2. **Start Small:**
    
    - Begin with low settings like `512 x 512` and `24 frames` to confirm the pipeline works.
        
    - Then scale up gradually.
        
3. **Length Optimization:**
    
    - For looping videos, you can generate a short clip and then repeat or interpolate it in post-processing.
        
4. **Batch Size = 1 (Recommended):**
    
    - Start with 1 to avoid VRAM overload.
        
    - Only increase if you have a powerful GPU (like 48GB VRAM).
        

---

## **Why This Node is Important**

Without this node, the KSampler wouldn’t have a latent video structure to work with. It’s basically like giving the model a blank canvas — but in 3D (time + width + height).

---

## **Summary Table**

|Parameter|Description|Typical Value|
|---|---|---|
|**width**|Frame width (pixels)|512, 640, 768|
|**height**|Frame height (pixels)|512, 640, 768|
|**length**|Number of frames in video|24–96|
|**batch_size**|Number of separate videos generated at once|1|
