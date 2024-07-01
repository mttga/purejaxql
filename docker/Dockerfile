FROM nvcr.io/nvidia/jax:23.10-py3

# Set the default build argument to requirements/requirements.txt
ARG REQUIREMENTS_FILE=requirements/requirements.txt

COPY ${REQUIREMENTS_FILE} requirements.txt
RUN pip install -r requirements.txt

# put your wandb api key here
#ENV WANDB_API_KEY= your api key here

# run stuff as non-root, comment if you want to run as root
ARG UID
ARG USERNAME
RUN useradd -u $UID --create-home $USERNAME
USER $USERNAME