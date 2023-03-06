#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 source_dir destination_dir"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "Error: $1 is not a directory"
    exit 1
fi

mkdir -p "$2"

cp -R "$1"/* "$2"
