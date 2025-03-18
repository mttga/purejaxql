gpu=1 # you can set this to 0,1,2 for multiple gpus or "all" for all gpus
docker run -d --rm --gpus=\"device=${gpu}\" --ipc=host \
 -v $(pwd):/app \
 -w /app \
 --name "pqn-${gpu//,/-}" \
 pqn \
 bash # python purejaxql/pqn_vdn_rnn_jaxmarl.py +alg=pqn_vdn_rnn_smax
 #python purejaxql/pqn_craftax.py +alg=pqn_craftax
 #python purejaxql/pqn_vdn_rnn_jaxmarl.py +alg=pqn_vdn_rnn_smax
