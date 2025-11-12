gpu=0 # you can set this to 0,1,2 for multiple gpus or "all" for all gpus
docker run -it --rm --gpus=\"device=${gpu}\" --ipc=host \
 -v $(pwd):/app \
 -w /app \
 --name "pqn-${gpu//,/-}" \
 pqn \
 bash #python purejaxql/pqn_gymnax.py +alg=pqn_cartpole