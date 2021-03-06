RELEASESDIR=~/.ai2thor/releases
FNAME=thor-201903131714-Linux64
mkdir $RELEASESDIR
wget http://s3-us-west-2.amazonaws.com/ai2-thor/builds/$FNAME.zip -P /tmp
unzip /tmp/$FNAME.zip -d /tmp/$FNAME/
mv /tmp/$FNAME/ $RELEASESDIR/$FNAME/
chmod +x $RELEASESDIR/$FNAME/$FNAME 

apt install -y xvfb
pip3 install ai2thor

# Fix matplotlib backend after installing xvfb
mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc

xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python3

