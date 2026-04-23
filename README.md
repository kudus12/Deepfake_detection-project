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
------------------------------------------------------------------------------------------------------------------------------------------
### Week 5 – CNN Improvement and Model Training

During Week 5, the CNN model was improved by adding more convolution and pooling layers to increase the model’s ability to learn deepfake features from video frames. GPU acceleration using CUDA was configured, and the learning rate and batch size were adjusted to improve training. An 80/20 train-validation split was also implemented. After training for multiple epochs, the model improved from around 50% accuracy to approximately 90% validation accuracy. The best performing model was saved automatically as `best_model.pth`.
------------------------------------------------------------------------------------------------------------------------------------------
### Week 6 – Inference Pipeline and Flask Integration

During Week 6, a dedicated `inference.py` file was developed so the trained model could be used for predictions. The system loads the saved CNN model, extracts frames from uploaded videos using OpenCV, resizes the frames to 224x224, normalises them, and converts them into tensors. Softmax was then used to calculate the prediction probability and return whether the video was Real or Fake with a confidence score. The model was also integrated into a Flask web application using `app.py`, allowing users to upload videos and view prediction results locally through the browser.
------------------------------------------------------------------------------------------------------------------------------------------
### Week 7 – Testing and Try-Out Model Experiment

During Week 7, the current Flask-based inference system was tested using real and fake videos from both inside and outside the dataset. The model worked overall, but some false positives and false negatives were still found. An “uncertain” prediction option was considered for low-confidence results, but it was not kept because it made the system more complicated without improving the results enough. A separate try-out version of the project was then created to test supervisor-suggested preprocessing ideas without affecting the main working system. The try-out model was trained from scratch and reached around 85% accuracy during testing.
------------------------------------------------------------------------------------------------------------------------------------------
### Week 8 – Continued Testing and Project Refinement

During Week 8, further testing and refinement were carried out on the deepfake detection system. The focus was on comparing the original model with the experimental try-out version and checking whether the new preprocessing ideas improved prediction results. The original system remained the stable version while the try-out version was used for experimentation. Testing continued on different real and fake video samples to identify cases where the model produced incorrect predictions.
------------------------------------------------------------------------------------------------------------------------------------------
### Week 9 – Final System Evaluation and Interface Improvement

During Week 9, the deployed Flask-based deepfake detection system was evaluated further using both real and fake videos. The results showed that the original baseline pipeline still performed better than the experimental try-out version, so the original stable version remained the final functional system. The web interface was also improved by showing the uploaded video after analysis and displaying the extracted frames used for inference. The layout and styling were updated to make the application look more professional and easier to use. Supervisor-suggested experimentation continued separately, but the main final system stayed based on the stronger original pipeline.
