# Stance Detection- 

### The project aims to detect stance of tweets and classify it as believer, denier or neutral for climate change. Our research aims to figure which major machine learning category, and more specifically which model or method is the best for stance detection.

- ### The following is an explanation of several important files and folders in our project:
	- **DataPreprocessing:** 
		- contains the code for preprocessing our dataset, as a note for preprocessing.py <br> it calls the functions in the helper.py file.
	- **Dataset:** 
		- Contains the tweet datasets from the Kaggle Dataset that contains the tweet ids As a note, after the preprocessing, we use the Preprocessed_Data for all the models. However we were able to improve all the metrics by around 2% with more data for the tensorflow model, so the Preprocessed_Data_Added_More.csv. 
	- **Datasets:**
	 	- This folder is empty, but this is were the original Kaggle climate dataset was stored.
	- **Google_drive_Large_files.txt:**
		- Since we could store the model or our orginal dataset from kaggle in Github we were able to save it in Google Drive and this file contains the links to it.
	- **LiveDemoDataset:**
		- This folder contains code and file so we could demontrate our GUI in our demo presentation.
	- **SemiSupervisedLearning:**
		- This folder contains the code and results for our models related to Semi-Supervised Learning for stance detection.
	- **SupervisedLearning:**
		- This folder contains the code and results for our models related to Supervised Learning for stance detection.
	- **TransferLearning:**
		- This folder contains the code and results for our models related to Transfer Learning for stance detection.
	- **UnsupervisedLearning:**
		- This folder contains the code and results for our models related to UnSupervised Learning for stance detection. 
	- **WebApp:**
		- This contains the code and files for our GUI of the Flask API and Front-End, which was shown in our demo presentation. Also, contains the graphs generated for our presentations and reports.