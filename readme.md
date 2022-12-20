# Hand to Text

## Background

Use openCV and mediapipe to create an application that converts ASL to text to create a proof-of-concept for an "speech to text" for ASL.
Current system is limited to the basic ASL alphabet and does not really do the more dynamic poses.

## Goals

Create an application that can detect ASL gestures live and type out what a user is signing.
Find a way to watch for dynamic poses.
Learn more about data science, machine learning, and ASL.

### What I've learned so far

- Review the problems that you've encountered and the tricks/resources you used to solve them
  List of struggles:

* Finding a way to properly store the data (both the inputs and the outputs)
* Static accuracy (optimizer and sigmoid output)
* Validation data (rounding, preventing overfitting)
* Gathering Data

### Future Development:

- Find a way to watch for dynamic poses (branch and develop it)
  - could store an inital pose and have that activate a listener for the next few poses (timer, store poses for that duration, analyze the array to see if the poses are in there -> denoise it)

## Development Instructions

### Setup

1. Clone the repository
2. Set up a virtual environment and activate it:
   Windows:

```
python3 -m venv venv
venv\Scripts\Activate
```

Mac:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements
   Windows:

```
python3 -m pip install opencv-python
python3 -m pip install mediapipe
python3 -m pip install tensorflow
python3 -m pip install pandas
```

4. Begin developing

### Data Collection

Use the DataCollector class to collect data either from a directory of images or live from the camera.
When camera is open press any key (except escape) to go to a new frame.
Press escape to save coordinates and input a letter value (caps only)
In terminal press crtl+c to exit data collection.

Coordinates are stored in an array of length n with shape (n, 21, 3). This is what is inputted into the model for training/predicting.
The values are set to a number value (A=1, Z=26), divided by 26, multiplied by 26 minus 1 and that value is the index of an array that is set to 1 (this is a redundant system please stop). This is what the outputs are trained on.

### Class Setup

#### DataCollector

This is the class that interacts with the camera, contains functions to record new data, do live feed, etc.

#### DataManager

This class manages data, contains functions to denoise data and format it in a way for it to be inputted into the model.

#### HandManager

This class is used to manage the model, contains train function, validate function, etc.
