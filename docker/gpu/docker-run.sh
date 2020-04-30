#!/usr/bin/env bash

nvidia-docker run -it -v /raid:/raid -v /nfs:/nfs --rm -ti --ipc=host --name cresi_v3 cresi_v3_image