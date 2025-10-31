| Command / Action | Description |
|------------------|-------------|
| ```docker run -itd --name my-container ubuntu bash``` | Start a container and keep it running in the background |
| ```docker exec -it my-container bash``` | Enter the container |
| `Ctrl + P, then Ctrl + Q` | Leave the container without stopping or deleting it |
| ```docker exec -it my-container bash``` | Re-enter the same container later |
| ```docker stop my-container && docker rm my-container``` | Stop and delete it when you’re done |
