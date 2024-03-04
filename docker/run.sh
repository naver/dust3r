#!/bin/bash

set -eux

# Check if docker is installed
if ! command -v docker &>/dev/null; then
    echo "Docker could not be found. Please install Docker and try again."
    exit 1
fi

# Check if docker-compose or docker compose is installed and set the appropriate command
if command -v docker-compose &>/dev/null; then
    dcomp="docker-compose"
elif command -v docker &>/dev/null && docker compose version &>/dev/null; then
    dcomp="docker compose"
else
    echo "Docker Compose could not be found. Please install Docker Compose and try again."
    exit 1
fi

# Parse command line arguments
with_cuda=0
for arg in "$@"; do
    case $arg in
        --with-cuda)
            with_cuda=1
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done

# Run the appropriate docker-compose file based on the --with-cuda flag
if [ "$with_cuda" -eq 1 ]; then
    $dcomp -f docker-compose-cuda.yml up --build
else
    $dcomp -f docker-compose-cpu.yml up --build
fi
