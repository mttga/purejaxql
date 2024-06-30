gpu=3 # you can set this to 0,1,2 for multiple gpus or "all" for all gpus
docker run -d --rm --gpus=\"device=${gpu}\" --ipc=host \
 -v $(pwd):/app \
 -w /app \
 --name "pqn-${gpu//,/-}" \
 pqn \
 python purejaxql/pqn_rnn_craftax.py +alg=pqn_rnn_craftax