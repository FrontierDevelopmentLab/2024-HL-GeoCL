Currently two modules are included in the src folder, namely DAGGER and SHEATH. 

Current implementation uses Python 3.10.0. To run the code, you need to install the required packages. You can do this by running the following command in the terminal:
```bash
pip install -r requirements.txt
```

In terms of contributing to the project, the project makes use of black for code formatting and isort for import sorting. Further PEP8 standards are enforced using flake8. To ensure that your code is formatted correctly, you can run the following commands in the terminal:
```bash
black .
isort --profile black .
flake8 --extend-ignore E203,E501,W605
```
Please fix any errors that are raised by these commands before submitting a pull request.