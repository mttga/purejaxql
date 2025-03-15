docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg USERNAME=$(whoami) \
    -f docker/atari.Dockerfile \
    -t pqn-atari \
    .