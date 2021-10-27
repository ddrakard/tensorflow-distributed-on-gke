FROM tensorflow/tensorflow:latest-gpu

WORKDIR /

RUN pip install --upgrade pip

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl && \
     rm -rf /var/lib/apt/lists/*

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install kubernetes

RUN mkdir /app

WORKDIR /app

COPY setup.py /app/setup.py
# This is required for setup.py to work
COPY README.md /app/README.md

RUN pip install .

COPY saved_weights /app/saved_weights
COPY distributed_training_transformer /app/distributed_training_transformer
COPY configuration /app/configuration

EXPOSE 4793

CMD [ "python3", "-u", "-m", "distributed_training_transformer" ]