
## **Main Purpose**

The **KSampler (Advanced)** node takes in a latent image and uses a diffusion process to:

- **Add noise and denoise** it.
    
- **Generate new images** from scratch based on prompts.
    
- **Refine or edit images** using partial denoising.
    

It’s basically the core part of image generation in ComfyUI, similar to the sampling step in Stable Diffusion.

---

## **Inputs and Parameters**

Here’s what each setting does:

### **Inputs**

1. **model** _(LATENT)_
    
    - The Stable Diffusion model you want to use.
        
    - Required for all generations.
        
2. **positive** _(Prompt)_
    
    - Your positive prompt, describing what you _want_ to see in the image.
        
3. **negative** _(Prompt)_
    
    - Your negative prompt, describing what you _don’t want_ to appear in the image.
        
4. **latent_image** _(Optional)_
    
    - Used for image-to-image workflows or partial denoising.
        
    - If left empty, the node will generate a new image from random noise.
        

---

### **Parameters**

These control the sampling process:

#### **1. add_noise**

- **disable / enable**
    
- If enabled, the node adds noise to the input latent image before denoising.
    
- Useful for img2img or inpainting workflows.
    
- **disable** = no extra noise added (used when you’re already working with a noisy latent).
    

---

#### **2. noise_seed**

- A number that controls randomness.
    
- **0** means fully random every time.
    
- Setting a specific seed makes results repeatable.
    

---

#### **3. control after generate**

- **fixed / random**
    
- Determines if the seed and noise stay constant between runs.
    
- **fixed** = stable, repeatable results.
    
- **random** = each generation will vary even with the same prompts.
    

---

#### **4. steps**

- Number of denoising steps (iterations).
    
- Higher = better quality, but slower.
    
- Typical range:
    
    - **Low quality / fast test:** 8–12
        
    - **Standard quality:** 20–30
        
    - **High detail:** 40+
        

---

#### **5. cfg (Classifier Free Guidance Scale)**

- Controls how strongly the model follows your prompt.
    
- Lower values (e.g., 3–5) = looser, more creative outputs.
    
- Higher values (e.g., 8–12) = strict adherence to prompt but can cause artifacts.
    

---

#### **6. sampler_name**

- Chooses the sampling algorithm:
    
    - **Euler** – balanced and fast.
        
    - **Euler A** – more random/creative.
        
    - **Heun, LMS, DPM++ 2M, etc.** – higher-quality but slower.
        
- **Euler** is a good default.
    

---

#### **7. scheduler**

- Decides how the noise is reduced step-by-step:
    
    - **simple** – default and reliable.
        
    - Other schedulers can slightly change the image’s look or speed.
        

---

#### **8. start_at_step**

- First step in the denoising process to begin at.
    
- Example:
    
    - If you set this to **4**, steps 0–3 will be skipped.
        
    - Useful for partial denoising or refining an image without fully regenerating it.
        

---

#### **9. end_at_step**

- The step at which denoising stops.
    
- Example:
    
    - If total steps = 12, and `end_at_step = 8`, it will only run up to step 8.
        

---

#### **10. return_with_leftover_noise**

- **enable / disable**
    
- If enabled, returns the partially noisy latent image **along with the final result**.
    
- Useful for chaining multiple samplers together.
    

---

## **Workflow Example**

Here’s how you’d typically use it:

- **Txt2Img:**
    
    - Leave `latent_image` empty.
        
    - `add_noise` = **enable** (adds noise to start from scratch).
        
    - Steps = **20–30**.
        
    - CFG = **7–9** for strong prompt guidance.
        
- **Img2Img:**
    
    - Provide a `latent_image`.
        
    - Set `add_noise` = **disable** (so you don’t over-noise it).
        
    - Adjust `start_at_step` and `end_at_step` to control how much of the image changes.
        
- **Inpainting or refinements:**
    
    - `start_at_step` = mid-range (like 4 out of 12).
        
    - Only denoise part of the latent to keep structure.
        

---

## **Summary Table**

|Parameter|Purpose|Typical Value|
|---|---|---|
|**add_noise**|Add initial noise|enable (txt2img)|
|**noise_seed**|Control randomness|0 or fixed number|
|**steps**|Total denoising steps|20–30|
|**cfg**|Prompt adherence strength|7–9|
|**sampler_name**|Sampling algorithm|Euler|
|**scheduler**|Step scheduling|simple|
|**start_at_step**|Where denoising begins|0 or 4|
|**end_at_step**|Where denoising ends|12|
|**return_with_leftover_noise**|Return extra noisy output|disable|

---

## **Key Tips**

- If you want **repeatable results**, set both:
    
    - `noise_seed` to a fixed number.
        
    - `control after generate` to **fixed**.
        
- If your images are **too random or messy**, lower `cfg` or increase `steps`.
    
- For **img2img edits**, set `add_noise` to **disable**, and carefully tune `start_at_step`.
    
