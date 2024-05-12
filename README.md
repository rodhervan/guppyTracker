# guppyTracker

This repository constains the code necessary for tracking guppy fishes and obtaining their position, distance and velocity.

## Getting started

The following steps indicate how to set up the code on a local machine.

### Prerequisites

- Python 3.11.5 or higher
- pip package manager

### Installation

1. It is highly recommended to create a virtual environment (although optional), for this make sure venv is installed or install it from terminal
```bash
   pip install virtualenv
```
Create the virtual environment and activate it
```bash
   python -m venv guppyTracker
   guppyTracker\Scripts\activate
```
2. Clone this repository
```bash
   git clone https://https://github.com/rodhervan/guppyTracker
```
3. Navigate to the project directory
```bash
   cd  path/to/project/guppyTracker
```
- (OPTIONAL) If using a CUDA GPU to speed up the inference time, make sure to install the corresponding pytorch library. For example for CUDA version 11.8 execute the command:
```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
You can check the different versions and options at https://pytorch.org/get-started/locally/

- Install the other prerquisite libraries
```bash
   pip install -r requirements.txt
```

## Usage

### Naming convention and files

