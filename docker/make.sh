#!/usr/bin/env bash
cp Dockerfile ../.
cd ../.
docker build --rm=true -t ga63fiy/genetic-snake-ai:0.0.1 .
rm Dockerfile
