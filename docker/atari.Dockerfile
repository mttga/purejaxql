FROM nvcr.io/nvidia/jax:23.10-py3

# run stuff as non-root, comment if you want to run as root
ARG UID
ARG USERNAME
RUN useradd -u $UID --create-home $USERNAME
USER $USERNAME

# install 
WORKDIR /app/
COPY . .

RUN pip install -e .[atari] 

# a specific version of blinker to work with this nvidia jax image
RUN pip install blinker==1.4.0

# put your wandb api key here
ENV WANDB_API_KEY=""