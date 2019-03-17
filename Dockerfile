FROM kulhanek/deep-rl-pytorch

#Install dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
  apt-get -y install xvfb libgl1-mesa-dri libglapi-mesa libosmesa6 mesa-utils \
  && rm -rf /var/lib/apt/lists/*
#build-essential libxi-dev libglu1-mesa-dev libglew-dev libvips \

# Fix agg after installing xvfb
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc

# Add minos
ADD https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 /usr/bin/dumb-init
RUN chmod 0777 /usr/bin/dumb-init
COPY install_minos.sh /tmp/install_minos.sh
RUN chmod +x /tmp/install_minos.sh && /tmp/install_minos.sh && rm /tmp/install_minos.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--", "xvfb-run", "-s", "-ac -screen 0 1280x1024x24"]