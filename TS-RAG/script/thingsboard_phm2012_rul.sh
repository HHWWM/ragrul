#!/usr/bin/env bash
set -e
python -m bearing.thingsboard_mqtt --config ./configs/phm2012_rul.yaml --bearing_dir "$1"
