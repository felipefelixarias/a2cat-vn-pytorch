FROM tensorflow/tensorflow
RUN sudo apt-get install openjdk-8-jdk & \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list & \
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - & \
    sudo apt-get update && sudo apt-get install bazel
RUN mkdir /tmp & cd /tmp & git clone https://github.com/deepmind/lab & cd lab
RUN mkdir /app & cd /app
COPY src /app
ENTRYPOINT ["/bin/python3"]
