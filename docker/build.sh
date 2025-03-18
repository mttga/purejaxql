docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg USERNAME=$(whoami) \
    -f docker/Dockerfile \
    -t pqn \
    .