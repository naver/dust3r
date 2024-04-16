#!/bin/bash

set -eux

# Default model name
model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

check_docker() {
    if ! command -v docker &>/dev/null; then
        echo "Docker could not be found. Please install Docker and try again."
        exit 1
    fi
}

download_model_checkpoint() { 
    if [ -f "./files/checkpoints/${model_name}" ]; then
        echo "Model checkpoint ${model_name} already exists. Skipping download."
        return
    fi
    echo "Downloading model checkpoint ${model_name}..."
    wget "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/${model_name}" -P ./files/checkpoints
}

set_dcomp() {
    if command -v docker-compose &>/dev/null; then
        dcomp="docker-compose"
    elif command -v docker &>/dev/null && docker compose version &>/dev/null; then
        dcomp="docker compose"
    else
        echo "Docker Compose could not be found. Please install Docker Compose and try again."
        exit 1
    fi
}

run_docker() {
    export MODEL=${model_name}
    if [ "$with_cuda" -eq 1 ]; then
        $dcomp -f docker-compose-cuda.yml up --build
    else
        $dcomp -f docker-compose-cpu.yml up --build
    fi
}

with_cuda=0
for arg in "$@"; do
    case $arg in
        --with-cuda)
            with_cuda=1
            ;;
        --model_name=*)
            model_name="${arg#*=}.pth"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done


main() {
    check_docker
    download_model_checkpoint
    set_dcomp
    run_docker
}

main
