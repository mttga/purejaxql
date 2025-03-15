FROM nvcr.io/nvidia/jax:23.10-py3

# install 
WORKDIR /app/
COPY . .

RUN pip install -e .[atari] 

# a specific version of blinker to work with this nvidia jax image
RUN pip install blinker==1.4.0

# put your wandb api key here
ENV WANDB_API_KEY=""