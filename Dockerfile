FROM kulhanek/deep-rl-pytorch

#Install dependencies
RUN apt-get update && \
  apt-get -y install build-essential libxi-dev libglu1-mesa-dev libglew-dev libvips \
  xvfb \
  && rm -rf /var/lib/apt/lists/*

# Fix agg after installing xvfb
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc

# Add minos
COPY install_minos.sh /tmp/install_minos.sh
RUN chmod +x /tmp/install_minos.sh && /tmp/install_minos.sh && rm /tmp/install_minos.sh

# Add ai2thor
RUN pip3 install ai2thor