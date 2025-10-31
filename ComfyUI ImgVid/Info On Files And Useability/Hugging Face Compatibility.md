
## **How to Tell if a HuggingFace Model Works with ComfyUI**

ComfyUI is built for Stable Diffusion and similar model types, so compatibility mainly depends on the **file format** and **model type**. Here’s what to look for:

### **1. Check the File Format**

- **Compatible file extensions:**
    
    - `.safetensors` → **Most recommended**, safer and faster loading.
        
    - `.ckpt` → Legacy Stable Diffusion checkpoints, still supported.
        
    - `.vae.pt` or `.vae.safetensors` → Separate VAE files.
        
    - `.lora` or `.safetensors` (LoRA) → For lightweight fine-tunes.
        

If you don’t see one of these, ComfyUI likely **can’t use it directly**.

---

### **2. Check the Model Type**

On HuggingFace, the description usually tells you what kind of model it is. Look for these keywords:

|**Keyword**|**Meaning**|**ComfyUI Use?**|
|---|---|---|
|`stable-diffusion`|A Stable Diffusion checkpoint (1.5, 2.1, XL, etc.)|✅ Yes|
|`vae`|Variational Autoencoder file|✅ Yes (optional)|
|`lora`|Lightweight fine-tune file|✅ Yes|
|`controlnet`|ControlNet model for pose, depth, etc.|✅ Yes|
|`text-to-image`|General AI text-to-image model|✅ Usually|
|`diffusers`|HuggingFace pipeline version|⚠ Needs conversion|
|`transformers`|Language or non-image models|❌ No|

> **Tip:**  
> If it says _"Requires diffusers pipeline"_ or there are no `.ckpt` or `.safetensors` files, you’ll likely need to **convert it first** before using it in ComfyUI.

---

### **3. Look for "Stable Diffusion" in the Tags**

At the top of a HuggingFace page, there are usually tags like:

> `stable-diffusion`, `text-to-image`, `vae`

If you don’t see `stable-diffusion`, it might not be a direct drop-in model for ComfyUI.

---

### **4. Check the Files Tab**

- Go to the **"Files and versions"** tab on HuggingFace.
    
- Look for files like:
    
    - `model.safetensors` or `model.ckpt`
        
    - `vae.safetensors`
        
    - `lora.safetensors`
        
    - `controlnet.pth`
        

If all you see are folders like `diffusers/` or just text files like `config.json`, that’s a sign you **need to convert it**.

---

## **Nodes in ComfyUI**

In ComfyUI, everything is built from **nodes**, like puzzle pieces that connect together. Each node does a single job.

### **Types of Nodes:**

|**Node Type**|**What It Does**|
|---|---|
|**Loader Nodes**|Load files like checkpoints, VAEs, LoRAs.|
|**Prompt Nodes**|Input text prompt and negative prompt.|
|**Processing Nodes**|Modify or process the image (upscale, filters).|
|**Sampler Nodes**|Generate the image using the model.|
|**Output Nodes**|Save or display the final image.|

For example, a very simple workflow:

1. **Checkpoint Loader** → loads your `.safetensors` model.
    
2. **Prompt Node** → where you type your prompt.
    
3. **Sampler Node** → runs the Stable Diffusion process.
    
4. **Save Image Node** → saves the final result.
    

---

## **Summary**

- **Look for `.safetensors` or `.ckpt` files** on HuggingFace.
    
- **Confirm it’s a Stable Diffusion model** (`stable-diffusion` tag is a good sign).
    
- **Nodes** are just individual steps in your image workflow, like loading a model or saving an image.
    
- If the HuggingFace page only has `diffusers/` files and no checkpoint, you’ll need to **convert it** before using it with ComfyUI.
