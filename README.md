# PyTorchMetalDemo

PyTorchMetalDemo is a demonstration project showcasing how to use Apple's Metal API with PyTorch to perform custom tensor operations on macOS devices with MPS (Metal Performance Shaders) support.

## Features

- Implements a custom tensor fill operation using Metal.
- Implements a custom tensor add operation using Metal.
- Demonstrates integration of Metal with PyTorch.

## Requirements

- macOS with Metal support
- PyTorch with MPS backend support
- Xcode command line tools
- Python 3.12
- uv package manager (https://github.com/astral-sh/uv) or any other python package manager

## Metal CPP

The metal-cpp package was downloaded from [here](https://developer.apple.com/metal/cpp/#:~:text=1.%20Prepare%20your%20Mac.). If you are not using macOS 15, please download the appropriate version from [this link](https://developer.apple.com/metal/cpp/#:~:text=1.%20Prepare%20your%20Mac.).

## Usage

``` bash

   git clone https://github.com/JohanLi233/PyTorchMetalDemo.git
   
   cd PyTorchMetalDemo

   # create venv
   uv venv --python=3.12
   source .venv/bin/activate

   uv pip install -r requirements.txt
   
   uv pip install -e .
   
   python test.py
```
