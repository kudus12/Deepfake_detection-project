# Deepfake Detection Project
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WEEK-1: - Progress Report
Tasks completed this week: -
•	Identified Face Forensics++ as the primary dataset for training and evaluation.
•	Downloaded Face Forensics++, which is about 20GB.
•	Reviewed Dataset structure, file formats, and class labels (real vs manipulated).
•	Updated the project plan using a Jira Kanban board with weekly milestones and time estimates.
 
Outcome: -

•	Dataset requirements are clearly understood.

•	Project timeline is structured and realistic.

•	The system is ready to move into image preprocessing next week.

Planned work for next week: - 

•	Begin image preprocessing (resizing, normalisation, format conversion).

•	Implement preprocessing pipeline in Python for PyTorch compatibility.


Planned work for next week-2: -

•	Begin image preprocessing (resizing, normalisation, format conversion).

•	Implement preprocessing pipeline in Python for PyTorch compatibility.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Week-2 Report – Image/video Preprocessing (Code-based)

This week, I prepared the FaceForensics++ dataset for use in the deepfake detection system. I classified the dataset into two categories: Real (original) and fake (deepfake altered). I wrote a Python preparation script with OpenCV to ensure that all video files could be opened and decoded appropriately. The script loops over the videos in both directories, reading the first frame and displaying frame characteristics like resolution and colour channels. This phase is critical because machine-learning algorithms require reliable and consistent input data. The findings indicated that both real and fake videos are accessible and suitable for further processing, implying that the dataset is ready for frame extraction and model training in the following steps.
Work Completed

•	Organised the FaceForensics++ dataset into genuine and fake folder structures.

•	Created a Python + OpenCV preparation tool for testing video files

•	I opened many videos and successfully decrypted the first frame.

•	Checked frame resolution and colour channel information (RGB).

•	Verified that actual and false videos are authentic and usable for machine learning training.

•	confirmed that the dataset is ready for frame extraction and CNN model training.

## Project Structure

- ML/ - Machine learning preprocessing and dataset testing scripts
- templates/ - HTML web interface templates
- static/ - CSS and static files
- app.py - Flask application

## Dataset Note
The real and fake deepfake datasets are very large and are stored locally.  
A `.gitignore` file is used to ensure datasets are NOT uploaded to GitHub.  
Only source code and system implementation files are included in this repository.

Planned work for next week-3: -

•Load the full FaceForensics++ dataset into PyTorch using DataLoader

•Prepare images (resize, normalise, and label real vs fake)
•Set up the basic CNN model structure for deepfake detection


-----------------------------------------------------------------------------------------------------------------------------
Report for Week 3: Model Design and Configuration

In Week 3, the emphasis was on building up the data pipeline and the fundamental model architecture using PyTorch in order to get the deepfake detection system ready for training. Real and deepfake films were loaded from the prepared dataset directories using a specially designed PyTorch dataset loader. 

The loader reads video files, extracts a predetermined number of frames from each video, normalizes pixel values, and resizes the videos to a standard resolution. Because each video is labeled as authentic or phony, the data can be used appropriately for supervised learning. This stage verifies that the dataset can be properly loaded and processed in a format compatible with deep learning

-----------------------------------------------------------------------------------------------------------------------------
## Project Weekly Progress

### Week 4 – Model Training Setup

During Week 4, the main focus was setting up the first training pipeline for the deepfake detection model. A PyTorch training script was created using a CNN model, CrossEntropyLoss for Real/Fake classification, and the Adam optimiser for updating the model weights. The training pipeline was tested using pre-processed video frame batches, and the input/output shapes were checked to make sure the model was working correctly. Initial forward and backward passes were completed successfully, and the loss values started to decrease, showing that the model was learning.

### Week 5 – CNN Improvement and Model Training

During Week 5, the CNN model was improved by adding more convolution and pooling layers to increase the model’s ability to learn deepfake features from video frames. GPU acceleration using CUDA was configured, and the learning rate and batch size were adjusted to improve training.
