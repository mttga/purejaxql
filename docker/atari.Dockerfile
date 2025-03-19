# envpool it's not compatible with newer versions of jax
FROM nvcr.io/nvidia/jax:24.04-py3

# run stuff as non-root, comment if you want to run as root
ARG UID
ARG USERNAME
RUN useradd -u $UID --create-home $USERNAME
USER $USERNAME

# install 
WORKDIR /app/
COPY . .

RUN pip install -e .[atari] 

# put your wandb api key here
ENV WANDB_API_KEY=""