
clone GitHub
git clone git@github.com:<repo location>

clone hugging face
git clone git@hf.co: <repo location>

start docker container (first time)
jetson-containers run --name <name> -v $(pwd):/workspace $(autotag <ollama/pytorch/etc...>) bash

Save docker container  
----

start docker container <from img>
jetson-containers run --name tinyllama -v $(pwd):/workspace reen16/jetson-tinyllama-1.1b:latest bash

enter a docker container
docker exec -it <container_id_or_name> bash

exit and shutdown conatiner
exit

exit and keep running
ctrl+p  & ctrl+q