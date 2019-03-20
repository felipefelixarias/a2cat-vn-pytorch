Bootstrap: docker
From: kulhanek/target-driven-visual-navigation:latest

%copy
jobs/container-inside.sh /runscript.sh

%runscript
exec /bin/bash runscript.sh "$@"