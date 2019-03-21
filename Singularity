Bootstrap: docker
From: kulhanek/target-driven-visual-navigation:latest

%runscript
echo "Installing new version of deep-rl"
pip3 install --user git+https://github.com/jkulhanek/deep-rl-pytorch.git
echo "Verifying mounted repository"
if [ -e /experiment ]
then
    echo "Experiment directory mounted"
    echo "Container is ready!"
    echo "Launching experiment with arguments [$@]"
    exec python3 "/experiment/$@"
else
    echo "You have to mount your repository to /experiment"
fi