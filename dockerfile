# deepmind-lab package container
FROM tensorflow/tensorflow:latest-py3 as artefact-builder
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
  build-essential \
  zip

RUN mkdir /toolbox && cd /toolbox &&  git clone https://github.com/jkulhanek/lab.git
RUN echo "#!/bin/bash\nrm /tmp/.X1-lock /tmp/.X11-unix/X1 \nvncserver :1 -randr -geometry 1280x800 -depth 24 && tail -F /root/.vnc/*.log" | tee /opt/vnc.sh
ENV DISPLAY :1

RUN cd /toolbox/lab && bazel build :deepmind_lab.so --define headless=osmesa --force_python=PY3 --host_force_python=PY3

# we will build the pip package
RUN cd /toolbox/lab && \
  bazel build -c opt python/pip_package:build_pip_package --force_python=PY3 --host_force_python=PY3 && \
  mkdir /artifact && \
  ./bazel-bin/python/pip_package/build_pip_package /artifact

RUN pip install /artifact/DeepMind_Lab-1.0-py3-none-any.whl

# Output container
FROM nvidia/cuda:9.0-base

RUN apt-get update && apt-get install -y \
  lua5.1 \
  liblua5.1-0-dev \
  libffi-dev \
  gettext \
  freeglut3-dev \
  libsdl2-dev \
  libosmesa6-dev \
  python3 \
  python3-pip \
  python3-dev \
  python3-numpy \
  realpath \
  build-essential \
  zip \
  git

RUN pip3 install --user --upgrade tensorflow-gpu


# Redirect display
RUN echo "#!/bin/bash\nrm /tmp/.X1-lock /tmp/.X11-unix/X1 \nvncserver :1 -randr -geometry 1280x800 -depth 24 && tail -F /root/.vnc/*.log" | tee /opt/vnc.sh
ENV DISPLAY :1

# Install package
COPY --from=artefact-builder /artifact/DeepMind_Lab-1.0-py3-none-any.whl /tmp/DeepMind_Lab-1.0-py3-none-any.whl
RUN pip3 install /tmp/DeepMind_Lab-1.0-py3-none-any.whl

# Install gym and other packages
RUN pip3 install gym && \
  pip3 install git+https://github.com/jkulhanek/gym-maze-env.git