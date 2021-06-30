
# HAND SIGN PREDICTIONS WITH MP HOLISTICS AND LSTM

## Acknowledgements

I hereby like to thank Mr. Nicholas Renotte for his amazing work on hand sign predictions. This project was an amazing journey of learning and I have enclosed some of the links which have helped me during my journey to make this project.
- [Nicholas Renotte](https://github.com/nicknochnack)

- [Media Pipe Holistic](https://ai.googleblog.com/2020/12/mediapipe-holistic-simultaneous-face.html)

- [LSTM](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
- [ASL(American Sign Language)](https://www.nidcd.nih.gov/health/american-sign-language#:~:text=American%20Sign%20Language%20(ASL)%20is,grammar%20that%20differs%20from%20English.&text=It%20is%20the%20primary%20language,many%20hearing%20people%20as%20well.)


## Inspiration and Overview

Mute people often come across scenarios where they face a huge gap while communication with normal people. There are some stae of the art models availabe which have taken an
approach on this problem by using <a href="https://www.mdpi.com/1424-8220/19/24/5429/htm">3DCNN and LSTM with FSM Context-Aware Model</a>and many more.<br>
The general concept is that a number of CNN layers are used followed by a number of LSTM layers, use of a pretrained mobile net followed by a number of LSTM layers. These models end up requiring large amounts of data to produce good results and also demand very high compute power due to presence of 30 to 40 million parameters.

## Advantages of using Mediapipe Holistic with LSTM(Long Short Term Memory)
1) **Less data** is required to produce a **hyper accurate model**.
2) Due to use of a much ***denser neural network**, it is **faster to train**.
3) As a result of above point from **30 to 40 million parameters** we are able to work with **half million parameters** and **get** the requisite **results**. 
4) As the neural network is simple the **detections** are much **faster**.

## Minimum Requirements
1) RAM : 8 GB and above
2) Disk Space : 2 GB is the approx size of the repository
3) Processor : i3  and above in 10th gen (for anything less than 10th Gen minimum Intel i5 is required)
4) GPU : Its good to have one.
5) CUDA and CUDNN : If GPU is available

## Setup Process
<br />
<b>Step 1.</b> Clone this repository: <a href = "https://github.com/VaibhavSaran/Hand-Sign-Prediction-with-MP-Holistics">Hand Sign Prediction</a>.
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv tfod
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source tfod/bin/activate # Linux
.\tfod\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfodj
</pre>
<br/>
<b>Step 5.</b> Run Jupyter Notebook or Jupyter Lab which ever is available. Run the command : 
<pre>
Jupyter Notebook #To run Jupyter Notebook 
# or run the below to launch Jupyter Lab
Jupyter Lab
</pre>
Error: If the above command is not recognized then try installing the module with command. Anyone will do or both can be done.
<pre>
pip install jupyterlab # To install Jupyter Lab
pip install jupyter # To install Jupyter Notebook
</pre>
<br/>
<b>Step 6.</b> After launching Jupyter make sure to select the kernel to the virtual environment.
<img src="https://i.imgur.com/8yac6Xl.png">
<br/>
<b>Step 7.</b> First run the notebook Dependencies Installation.ipynb, it will install all the reuisite libraries for your system. However do check which version of tensorflow you are installing and the corresponding CUDA and CUDnn libraries.

## Executing Project
Run all the cells of the notebook **Handsign Prediction.ipynb**. It has a separate cell to collect data which can be skipped if you dont want to train model from scratch rather use pretrained weights. The model has been saved and is available as HSD_Ver1.h5 which can imported in the notebook with the script <pre> model.load("HSD_Ver1.h5").</pre>  

## Working Summary of Project
1) Collecting keypoints from MediaPipe Holistic.
2) Training a Deep Neural Network with LSTM layers for sequences.
3) Performing real time sign language detection using openCV.

## Data Collection
For this project I have not used any data from any 3rd party source/company. I went and created my own data by taking videos of myself performing the signs, applying keypoints to converting them to numpy arrays for model training. Everything in this project has been implemented from scratch and due to less amount of data available, model might seem to be a bit janky but with adequate data it can perform very well.

## In Depth Project Steps and Working
<b>Step 1.</b> The first step is of importing all the required dependencies and verifying the status of CUDA libraries and whether tensorflow is using GPU or not.
<br/>
<b>Step 2.</b> In the second point Mediapipe holistic is setuped to generate keypoints and draw the landmarks on face, pose, left hand and right hand.
<img src="https://i.imgur.com/kyOf4XV.png">
(one hand is missing as it was used to quit the screen.)
<br/>
<b>Step 3.</b> The keypoints which were generated will be needed to be tracked  and extracted. In total for this project 1662 keypoints are being used so to track and extract them we define a function for the same.
<br/>
<b>Step 4.</b> Folders and paths for the collection of the keypoints are setup using the OS library for each of action which is "Hello" , "Thank You" and "I Love You".
<br/>
<b>Step 5.</b> In this step we collect keypoints for each of the handsign in video format. A total of 30 Videos of size 30 frames for 3 handsigns and in each frame there are 1662 keypoints which are being tracked and stored.
<br/>
<b>Step 6.</b> In this step the collected data is preprocessed, Labels are created for each action and features are created. As a result the input has 90 videos with 30 frames in each of those videos with 1662 values which represent our keypoints. The output y is one hot encoded and we get 90 labels in the shape of (90,3).

A sub part if this step is partition of data into train and test. The reason for taking only 5% of the data for test is because of lack of available data for each action, so if we use a standard test train split of 70-30 or 60-40 etc. Due to lack of available data the LSTM model will not be able to train properly and will not perform well in real time predictions. NOTE: In case we have more data while executing this notebook we can increase the test size to a standard partition if there is adequate data available.
<br/>
<b>Step 7.</b> In this step the LSTM Model is built with the architecture shown in picture.
<img src = "https://i.imgur.com/OSQF0Mn.png">
A summary of the LSTM Model:
<img src ="https://i.imgur.com/Sy3cAok.png">
The final number of trainable parameters indicate half a million parameters which is very less comapred to 30 million parameters used in other state of the art models.
<br/>
<b>Step 8.</b> In this step we try to make static predictions on our model to confirm whether it is performing and predicting as we desired it.
<br/>
<b>Step 9.</b> After confirming that our model is performing well, we save the model in this step.
<br/>
<b>Step 10.</b> In this step model is evaluated to check how accurate it is, in the current situation it has very less data so the evaluation may not yield very optimal results but it still goes on to do great predictions and with addition of more data it will be even more effective.
<br/>
<b>Step 11.</b> In this step we perform predictions in real time from the feed which we get from the camera.

## Future Scope and Improvements
1) Adding more data to perform better predictions.
2) Adding more signs for recognition.
3) Integrating with ASL alphabets so that basic conversation can take place using this model.
4) It can be used to develop a Robot to recognize these hand signs. These robots can then be deployed at Airports and railway stations which can help in easing the communication between mute people and authorities.