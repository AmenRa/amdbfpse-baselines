#!/bin/bash

LIGHT_GREEN='\033[1;32m'
LIGHT_BLUE='\033[1;34m'
NO_COLOR='\033[0m'

# Create temporary folder
mkdir -p tmp

# Create dataset folder
mkdir -p datasets

# Download Computer Science dataset archive
printf "${LIGHT_BLUE}Starting Computer Science dataset archive download...\n${NO_COLOR}"
wget -c -O tmp/computer_science.zip https://zenodo.org/record/6606557/files/computer_science.zip
printf "${LIGHT_GREEN}Computer Science dataset archive download: DONE\n\n${NO_COLOR}"
# Decompress Computer Science dataset archive
printf "${LIGHT_BLUE}Starting Computer Science dataset archive extraction...\n${NO_COLOR}"
7z x -o"datasets" tmp/computer_science.zip
printf "${LIGHT_GREEN}Computer Science dataset archive extraction: DONE\n\n${NO_COLOR}"

# Download Physics dataset archive
printf "${LIGHT_BLUE}Starting Physics dataset archive download...\n${NO_COLOR}"
wget -c -O tmp/physics.zip https://zenodo.org/record/6606557/files/physics.zip
printf "${LIGHT_GREEN}Physics dataset archive download: DONE\n\n${NO_COLOR}"
# Decompress Physics dataset archive
printf "${LIGHT_BLUE}Starting Physics dataset archive extraction...\n${NO_COLOR}"
7z x -o"datasets" tmp/physics.zip
printf "${LIGHT_GREEN}Physics dataset archive extraction: DONE\n\n${NO_COLOR}"

# Download Political Science dataset archive
printf "${LIGHT_BLUE}Starting Political Science dataset archive download...\n${NO_COLOR}"
wget -c -O tmp/poltical_science.zip https://zenodo.org/record/6606557/files/political_science.zip
printf "${LIGHT_GREEN}Political Science dataset archive download: DONE\n\n${NO_COLOR}"
# Decompress Political Science dataset archive
printf "${LIGHT_BLUE}Starting Political Science dataset archive extraction...\n${NO_COLOR}"
7z x -o"datasets" tmp/political_science.zip
printf "${LIGHT_GREEN}Political Science dataset archive extraction: DONE\n\n${NO_COLOR}"

# Download Psychology dataset archive
printf "${LIGHT_BLUE}Starting Psychology dataset archive download...\n${NO_COLOR}"
wget -c -O tmp/psychology.zip https://zenodo.org/record/6606557/files/psychology.zip
printf "${LIGHT_GREEN}Psychology dataset archive download: DONE\n\n${NO_COLOR}"
# Decompress Psychology dataset archive
printf "${LIGHT_BLUE}Starting Psychology dataset archive extraction...\n${NO_COLOR}"
7z x -o"datasets" tmp/psychology.zip
printf "${LIGHT_GREEN}Psychology dataset archive extraction: DONE\n\n${NO_COLOR}"