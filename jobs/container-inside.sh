#!/bin/bash
echo "Downloading new version of deep-rl-pytorch"
#pip3 install --user git+https://github.com/jkulhanek/deep-rl-pytorch.git
echo "Verifying mounted repository"
if [ -e /opt ]
then
    echo "Experiment directory mounted"
    echo "Container is ready!"
    echo "Launching experiment with arguments [$@]"
    exec python3 "/usr/bin/$@"
else
    echo "You have to mount your repository to /experiment"
fi