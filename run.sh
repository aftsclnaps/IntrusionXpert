#!/bin/bash

docker build -t intrusionxpert .
docker run -p 8501:8501 intrusionxpert