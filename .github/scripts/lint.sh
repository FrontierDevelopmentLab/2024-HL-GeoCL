#!/usr/bin/env bash
set -ex

black --check --diff .
isort --profile black --check --diff .
flake8 --extend-ignore E203,E501,W605
