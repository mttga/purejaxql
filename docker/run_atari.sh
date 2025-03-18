gpu=2
docker run -it --rm --gpus=\"device=${gpu}\" --ipc=host \
 -v $(pwd):/app \
 -w /app \
 --name "pqn-${gpu//,/-}" \
 pqn-atari \
 bash #python purejaxql/pqn_atari.py +alg=pqn_atari