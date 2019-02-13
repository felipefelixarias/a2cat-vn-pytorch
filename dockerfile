FROM tensorflow/tensorflow
RUN apt-get update && apt-get install -y curl openjdk-8-jdk git
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list & \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && apt-get install bazel -y

RUN apt-get update && apt-get install -y \
  lua5.1 \
  liblua5.1-0-dev \
  libffi-dev \
  gettext \
  freeglut3-dev \
  libsdl2-dev \
  libosmesa6-dev \
  python3-dev \
  python3-numpy \
  realpath \
  build-essential

RUN mkdir /toolbox && cd /toolbox && git clone https://github.com/deepmind/lab.git && cd lab
RUN echo "#!/bin/bash\nrm /tmp/.X1-lock /tmp/.X11-unix/X1 \nvncserver :1 -randr -geometry 1280x800 -depth 24 && tail -F /root/.vnc/*.log" | tee /opt/vnc.sh
ENV DISPLAY :1

RUN bazel build :deepmind_lab.so --define headless=osmesa