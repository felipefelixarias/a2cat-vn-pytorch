#!/bin/bash
mkdir /root/.minos && \
git clone https://github.com/minosworld/minos.git /root/.minos && \
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.7/install.sh | bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install v10.13.0 && \
  npm install --prefix /root/.minos/minos/server && \
  pip3 install -e /root/.minos -r /root/.minos/requirements.txt && \
  pip3 install -e /root/.minos/gym