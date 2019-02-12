FROM tensorflow/tensorflow
RUN apt-get install openjdk-8-jdk curl -y & \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list & \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - & \
    apt-get update && apt-get install bazel -y
RUN mkdir /tmp & cd /tmp & git clone https://github.com/deepmind/lab & cd lab
RUN mkdir /app & cd /app
COPY src /app
ENTRYPOINT ["/bin/python3"]
