FROM nvidia/cudagl:9.0-devel

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev
    # && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch \
    torchvision \
    visdom \
    opencv-python \
    opencv-contrib-python \
    matplotlib

#Install dependencies
ENV DEBIAN_FRONTEND=noninteractive 
# RUN apt-get update && \
#   apt-get -y install xvfb libgl1-mesa-dri libglapi-mesa libosmesa6 mesa-utils && \
#  rm -rf /var/lib/apt/lists/*

# Fix agg after installing xvfb
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc

# Add minos
# ADD https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 /usr/bin/dumb-init
# RUN chmod 0777 /usr/bin/dumb-init
# COPY install_minos.sh /tmp/install_minos.sh
# RUN chmod +x /tmp/install_minos.sh && /tmp/install_minos.sh && rm /tmp/install_minos.sh

# ENTRYPOINT ["/usr/bin/dumb-init", "--", "xvfb-run", "-s", "-ac -screen 0 1280x1024x24"]


RUN apt-get install -y \
	libglfw3-dev libglm-dev libx11-dev libegl1-mesa-dev \
  libsm6 libopenmpi-dev \
	libpng-dev libjpeg-dev \
	build-essential pkg-config git curl wget automake libtool \
  && rm -rf /var/lib/apt/lists/*

# tqdm is only used by the tests
RUN pip3 install tqdm


# update git
RUN git clone --recursive https://github.com/facebookresearch/House3D /House3D
RUN wget https://github.com/facebookresearch/House3D/releases/download/example-data/example-data.tgz \
 -O /House3D/data.tgz && cd /House3D && tar xzf data.tgz
ENV TEST_HOUSE /House3D/house/05cac5f7fdd5f8138234164e76a97383/house.obj

# build renderer
WORKDIR /House3D/renderer
ENV PYTHON_CONFIG python3-config
RUN make -j


# install House3D and baselines
WORKDIR /House3D
RUN pip3 install -e . && \
  pip3 install tensorflow && \
  pip3 install git+https://github.com/openai/baselines.git

WORKDIR /root