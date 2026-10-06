#!/usr/bin/env bash
# setup_project.sh - Sets up project directory tree on Linux/macOS
set -e

echo "Creating project directory structure..."
mkdir -p data/raw/d1_alibaba2018
mkdir -p data/raw/d2_azure2019
mkdir -p data/raw/d3_alibaba_gpu2020
mkdir -p data/parquet
mkdir -p data/processed
mkdir -p src
mkdir -p results/figures
mkdir -p results/tables

echo "Directory structure created successfully."
