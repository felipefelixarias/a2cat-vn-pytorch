# !/bin/bash

cat << EOF > /root/xorg.conf
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "$AI2THOR_DEVICE_BUSID"
EndSection
Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
Section "ServerLayout"
    Identifier     "Layout0"
    Screen 0 "Screen0" 0 0
EndSection
EOF


Xorg -noreset -logverbose -logfile xorg.log -config /root/xorg.conf :0&
sleep 1
# x11vnc -display :0 -rfbauth /root/.vnc/passwd&
cd /experiments/target-driven-visual-navigation/
DISPLAY=:0.0 python3 train.sh $EXPERIMENT