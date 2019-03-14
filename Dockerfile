FROM kulhanek/deep-rl-pytorch

#Install dependencies
RUN apt-get update && \
  apt-get -y install build-essential libxi-dev libglu1-mesa-dev libglew-dev libvips \
  && xvfb \
  && rm -rf /var/lib/apt/lists/*

# Fix agg after installing xvfb
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc

# Add minos
RUN mkdir /root/.minos && \
  git clone https://github.com/minosworld/minos.git /root/.minos && \
  curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.7/install.sh | bash && \
  source ~/.bashrc && \
  nvm install v10.13.0 && \
  npm install --prefix /root/.minos/minos/server && \
  pip3 install -e /root/.minos -r /root/.minos/requirements.txt && \
  pip3 install -e /root/.minos/gym

# Add ai2thor
RUN pip3 install ai2thor




