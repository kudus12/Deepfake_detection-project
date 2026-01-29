# Deepfake Detection Project
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

Planned work for next week: - 
•	Begin image preprocessing (resizing, normalisation, format conversion).
•	Implement preprocessing pipeline in Python for PyTorch compatibility.


This project implements a Flask-based web interface and machine learning preprocessing pipeline for detecting deepfake videos.

## Project Structure
- ML/ - Machine learning preprocessing and dataset testing scripts
- templates/ - HTML web interface templates
- static/ - CSS and static files
- app.py - Flask application

## Dataset Note
The real and fake deepfake datasets are very large and are stored locally.  
A `.gitignore` file is used to ensure datasets are NOT uploaded to GitHub.  
Only source code and system implementation files are included in this repository.
