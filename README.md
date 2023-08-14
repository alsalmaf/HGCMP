# HGCMP(Hand Gesture Controlled Music PLayer)
Welcome to the Hand Gesture Recognition for Music Player Control GitHub repository! This project is all about redefining how you interact with your music player. Say goodbye to physical buttons and traditional input methods – our program empowers you to control your music playback with the power of your hands.

## What Is Hand Gesture Recognition?
Hand Gesture Recognition is an exciting field of computer vision that enables machines to understand and interpret human hand movements. In this project, this technology has been harnessed to create an intuitive way to interact with your music player. By recognizing specific hand gestures, you can effortlessly control your music without touching a single button.


## Features and Functionality:

The product will offer the following features and functionality for controlling the music player using hand gestures:

### Play/Pause Track:
Users can make a "shaka" sign with their hand to play or pause the currently playing track. The "shaka" gesture is formed by joining the thumb and index finger to create a circle, while the remaining fingers are extended. . When the user makes this gesture, it will serve as the trigger to toggle between play and pause states.

### Volume Control:
Users can adjust the volume level using hand gestures with zero to five fingers. Each finger represents a specific volume level, ranging from 0 to 100%. For example:

- Zero fingers (closed fist) will mute the volume.
- One finger will set the volume to 20%.
- Two fingers will set the volume to 40%
- Three fingers will set the volume to 60%
- Four fingers will set the volume to 80%
- Five fingers (open hand) will set the volume to 100%.

### Playlist Navigation:
Users can use a thumbs gesture to navigate through the playlist. A thumbs-down gesture to the right will move to the next track, while a thumbs-up gesture to the left will go back to the previous track.

# How to Use

1. **Clone the Repository:** Start by cloning this repository to your local machine.
```
    git clone https://github.com/alsalmaf/HGCMP.git
    cd HGCMP
```

3. **Set Up Dependencies:** Install the required dependencies using the provided instructions.
       
   *The current latest version mediapipe 0.9.0.1 (a machine learning framework neccary for this project) currently provides wheels for Windows for Python 3.7-3.10 so you may need to downgrade your Python version.
```
    sh setup.sh
```

5. **Run the Program:** Execute the program, and your webcam will become your music control center.
```
    python run.py
```
