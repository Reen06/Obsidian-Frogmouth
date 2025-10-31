### **New repo workflow**
```powershell
# (1) Initialize a new Git repository
git init

# (2) Add all files in your current directory
git add .

# (3) Make your first commit
git commit -m "Initial commit"

# (4) Add your GitHub repo as the remote (using SSH)
git remote add origin git@github.com:Reen06/<repo>

# (5) Verify that the remote was added correctly
git remote -v

# (6) Push your files to GitHub
git branch -M main
git push -u origin main
```

### **Clone repo workflow**
```powershell
# (1) Initialize a new Git repository
git init

# (2) Add all files in your current directory
git clone git@github.com:Reen06/<repo>

```

### **Normal update workflow**

```powershell
git pull origin main          # get the latest version from GitHub
git add .                     # stage all your changes
git commit -m "Update"        # commit them locally
git push origin main          # upload to GitHub
```

---

### Notes:

- You **don’t need `--allow-unrelated-histories`** anymore — that’s only for the very first merge if your local and remote histories were different.
    
- You only need `-u` (the `-u` flag in `git push -u origin main`) once — after that, you can just type:
    
    ```powershell
    git push
    ```
    
    and Git will remember the branch to push to.
    

---

###  In short:

Next time you make changes, just do:

```powershell
git add .
git commit -m "your message"
git push
```

That’s it. GitHub will update your repo with whatever’s new in your local folder.