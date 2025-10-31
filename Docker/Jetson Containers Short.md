## **Jetson Containers Quick Reference**

### **1. Create Container with Current Folder Mounted**

Create a container with the current directory mounted to `/workspace`:

```bash
jetson-containers run --name <name> -v $(pwd):/workspace $(autotag ollama) bash
```

---

### **2. Exit Container**

**First exit (detach without stopping):**
- Press `Ctrl+P` then `Ctrl+Q`

**Subsequent exits (stop the container):**
```bash
exit
```

---

### **3. List Docker Containers**

|**Command**|**Description**|
|---|---|
|`docker ps`|List only running containers|
|`docker ps -a`|List all containers (running and stopped)|

---

### **4. List Docker Images**

```bash
docker images
```

---

### **5. Remove Unused Images**

```bash
docker image prune
```

---

### **6. Remove Containers**

```bash
docker rm <container_name_or_id>
```

To force remove a running container:

```bash
docker rm -f <container_name_or_id>
```

---

> **Tip:**  
> Use `docker ps -a` to see all containers before removing them, as containers must be stopped before removal (unless using `-f` flag).
