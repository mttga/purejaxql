FROM nvcr.io/nvidia/jax:25.01-py3

# for mujoco playground rendering
RUN apt-get update && apt-get install -y git libglfw3 libgl1 libosmesa6 cmake ffmpeg

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