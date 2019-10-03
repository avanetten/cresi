#!/usr/bin/env bash

nvidia-docker run -it -v /raid:/raid -v /nfs:/nfs --rm -ti --ipc=host --name cresi_v2 cresi_v2_image