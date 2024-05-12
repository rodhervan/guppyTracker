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

4. Install the other prerquisite libraries
```bash
   pip install -r requirements.txt
```

## Usage

### Loading a video

The file yoloPredict.py loads an `.mp4` video and extracts the position of the fish using a YOLOv8 segmentation model. A pretrained model for this project is located inside the Train_data folder; traning models end in a `.pt` extension. To set the start time to something different than cero change the variable `start_time_ms` to a different value in miliseconds. This code will initiate a window showing the video an will put a green polygon to indicate the outline of the fish; a blue circle will indicate the centroid of this polygon. This window will continue to reproduce the video until it ends; to stop the video playback press the letter `Q`. Once the video playback ends and `.JSON` file will be generated containing the data extracted from the video. 

#### JSON File structure

The `.JSON` file contains a list of dictionaries, where each item in this list corresponds to a single frame. In each frame the following data will be stored under the following keys:
´"frame_number"´: Contains an integer, where the first frame will start as 0 and increase by 1 for each frame.

