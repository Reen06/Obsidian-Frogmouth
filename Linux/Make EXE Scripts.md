## **Creating Executable Scripts on Linux**

### **1. Create the Script File**

Create a script file with a `.sh` extension (e.g., `script.sh`):

```bash
#!/bin/bash
cd /home/jetson/Projects/How-To || exit
git pull
frogmouth .
```

### **2. Make the Script Executable**

To make the script executable, run:

```bash
chmod +x ~/Projects/Jetson\ Scripts/Open-Notes.sh
```

### **3. Make the Script Globally Accessible**

To make the script available from anywhere in the terminal:

```bash
sudo mv "$HOME/Projects/Jetson Scripts/Open-Notes.sh" /usr/local/bin/notes
```

After this, you can run the script from any directory by typing `open-notes` in the terminal.

### **4. To edit the Script**

Run:

```bash
sudo micro /usr/local/bin/notes
```

---



> **Tip:**  
> The script name in `/usr/local/bin/` becomes the command you use. In this example, `Open-Notes.sh` becomes `open-notes`.
