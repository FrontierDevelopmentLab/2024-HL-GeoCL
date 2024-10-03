#!/bin/bash

# Run Black to format the code
echo "Running Black..."
black .

# Run isort to sort imports according to Black style
echo "Running isort..."
isort --profile black .

# Run flake8 to lint the code, ignoring specified errors
echo "Running flake8..."
flake8 --extend-ignore E203,E501,W605

echo "All tasks completed."
