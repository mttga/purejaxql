docker build \
    --build-arg REQUIREMENTS_FILE=requirements/requirements.txt \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg USERNAME=$(whoami) \
    -f docker/Dockerfile \
    -t pqn \
    .