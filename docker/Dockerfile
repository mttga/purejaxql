FROM nvcr.io/nvidia/jax:25.01-py3

# run stuff as non-root, comment if you want to run as root
ARG UID
ARG USERNAME
RUN useradd -u $UID --create-home $USERNAME
USER $USERNAME

# install 
WORKDIR /app/
COPY . .

RUN pip install -e .[jax_envs]

# put your wandb api key here
ENV WANDB_API_KEY=""