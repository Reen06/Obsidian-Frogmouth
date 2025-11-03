## **GitHub, Hugging Face & Docker Commands**

---

## **Git Operations**

### **Clone GitHub Repository**

```bash
git clone git@github.com:<repo_location>
```

---

### **Clone Hugging Face Repository**

```bash
git clone git@hf.co:<repo_location>
```

---

## **Docker Container Operations**

### **Start Docker Container (First Time - From Autotag)**

```bash
jetson-containers run --name <name> -v $(pwd):/workspace $(autotag <ollama/pytorch/etc...>) bash
```

---

### **Start Docker Container (From Image)**

```bash
jetson-containers run --name <name> -v $(pwd):/workspace <image_name>:<tag> bash
```

**Example:**
```bash
jetson-containers run --name tinyllama -v $(pwd):/workspace reen16/jetson-tinyllama-1.1b:latest bash
```

---

### **Enter a Docker Container**

```bash
docker exec -it <container_id_or_name> bash
```

---

### **Exit and Shutdown Container**

```bash
exit
```

---

### **Exit and Keep Container Running**

Press `Ctrl+P` then `Ctrl+Q`

---

### **Delete Docker Container**

```bash
docker rm <container_name>
```

---