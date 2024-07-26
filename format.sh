#!/usr/bin/env sh
black -l 120 -t py310 .
isort --profile black .
