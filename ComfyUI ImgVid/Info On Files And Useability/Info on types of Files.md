
[[Folder Structure.png]]

## **1. Checkpoints (Main Models)**

**What they are:**
These are the **core Stable Diffusion models** — basically the brain that generates images. They define the overall style, quality, and general knowledge of the AI.

* **Common formats:**
  `.safetensors` (preferred, safer) or `.ckpt` (older format).

**Examples of popular checkpoints:**

* `v1-5-pruned.safetensors` → Standard SD 1.5 base model.
* `realisticVision.safetensors` → More realistic photos.
* `dreamshaper.safetensors` → Artistic, flexible, good for creative styles.
* `sdxl.safetensors` → Newer SDXL model, higher quality.

**How to use in ComfyUI:**

* Place in `models/checkpoints/`
* In your workflow, add a **Load Checkpoint** node.
* Select your model from the dropdown.
* Everything you generate will use this as the foundation.

> **Tip:** Think of this like choosing which art engine you’re starting with — realistic, anime, stylized, etc.

---

## **2. VAE (Variational Autoencoder)**

**What they are:**
A **VAE fine-tunes how colors and details are handled**. Without a good VAE, your images might look washed out or have weird color issues.

* Some checkpoints **come with a built-in VAE**.
* Others need you to **load one manually**.

**Examples:**

* `vae-ft-mse-840000-ema-pruned.safetensors` → A solid all-around VAE.
* `orangemix.vae.pt` → Great for anime-style art.

**How to use:**

1. Put VAE files in `models/vae/`.
2. In ComfyUI, use a **VAE Encode/Decode** node or load it through the **Checkpoint Loader** if supported.
3. Try toggling between no VAE and a good VAE — you’ll *instantly* see richer colors and more natural lighting.

> **Tip:** If your final image looks "blurry" or "gray," you probably need a better VAE.

---

## **3. LoRA (Low-Rank Adaptation)**

**What they are:**
LoRAs are **lightweight add-ons** that can completely change or enhance your model’s output without replacing the main checkpoint.

* Think of them as "plug-ins" that add styles, characters, or specific looks.
* Much smaller than checkpoints (usually 50MB–200MB instead of gigabytes).

**Examples:**

* A LoRA for a specific anime character.
* A LoRA for making everything look like Pixar animation.
* A LoRA for realistic tattoos or cyberpunk gear.

**How to use:**

1. Put them in `models/lora/`.
2. In ComfyUI, use a **Load LoRA** node.
3. Connect it to your **main checkpoint** node.
4. Adjust the **weight (strength)**:

   * 0.7 → Strong effect.
   * 0.3 → Subtle touch.

> **Example:**
> Base model = `realisticVision.safetensors`
> LoRA = `cyberpunk_style.safetensors`
> Final image → realistic photo but with a cyberpunk vibe.

---

## **4. ControlNet Models**

**What they are:**
ControlNet lets you **guide the AI using external input**, like:

* A sketch you drew
* A pose reference
* Depth maps
* Edge detection (like outlines of an image)

This gives you **much more control** over composition and layout.

**Examples of ControlNet models:**

* `control_openpose.safetensors` → Controls poses with stick-figure skeletons.
* `control_depth.safetensors` → Uses depth maps for accurate 3D structure.
* `control_canny.safetensors` → Follows line art or outlines.

**How to use:**

1. Put them in `models/controlnet/`.
2. Add a **ControlNet Loader** node in ComfyUI.
3. Connect:

   * **Your input image/sketch** to the ControlNet.
   * ControlNet → Main checkpoint.
4. Generate → the AI will respect the structure you provided.

> **Example:**
> You upload a stick figure pose → AI outputs a full, detailed person **exactly in that pose**.

---

## **5. Upscalers (Image Enhancement Models)**

**What they are:**
These models **increase image resolution** and clean up details.

* Useful if your output is too small or slightly blurry.
* Often used at the end of a workflow to make images print-ready.

**Examples:**

* `RealESRGAN_x4plus.pth` → Standard upscaler.
* `4x-UltraSharp.pth` → Sharper details, great for portraits.

**How to use:**

1. Put them in `models/upscale_models/`.
2. Add an **Upscale** node in your workflow.
3. Choose your upscaler model.
4. Run it on your final output.

> **Example:**
> 512×512 → Upscale to 2048×2048 for a poster or high-quality print.

---

## **Workflow Example**

Here’s how they might all fit together:

```
Prompt → Checkpoint → LoRA → ControlNet (optional) → VAE Decode → Upscaler → Final Image
```

### Example Use Case:

> You want a high-res anime character in a specific pose:

1. **Checkpoint**: Anime base model (`anime_base.safetensors`).
2. **LoRA**: Add specific style or character details.
3. **ControlNet**: Guide with a stick figure pose.
4. **VAE**: Enhance colors and shading.
5. **Upscaler**: Make the image super sharp for wallpaper or print.


