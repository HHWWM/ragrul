#!/usr/bin/env bash
set -e
python -m bearing.preprocess_phm2012 --config ./configs/phm2012_rul.yaml
python -m bearing.retrieve_bearing --config ./configs/phm2012_rul.yaml
