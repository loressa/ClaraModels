export OPTIONS="--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"
export SOURCE_DIR=/home/ubuntu/aiaa_workspace
export MOUNT_DIR=/aiaa_workspace
export HOST_PORT=80
export DOCKER_PORT=5000
#export LOCAL_PORT=5000
#export REMOTE_PORT=80
#export DOCKER_IMAGE="nvcr.io/nvidia/clara-train-sdk:<version here>"
export DOCKER_IMAGE="nvcr.io/ea-nvidia-clara-train/clara-train-sdk:v4.0-EA2"

docker run $OPTIONS --gpus=1 -it --rm \
       -d --name aiaa-server \
       -p $HOST_PORT:$DOCKER_PORT \
       -v $SOURCE_DIR:$MOUNT_DIR \
       --ipc=host \
       $DOCKER_IMAGE \
       start_aiaa.sh --workspace /aiaa_workspace/aiaa-1 &
