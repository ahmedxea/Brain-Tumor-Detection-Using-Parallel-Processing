# Brain Tumor Detection

This project aims to detect brain tumors from MRI images using machine learning and parallel processing to enhance efficiency. Multiprocessing is implemented to optimize image preprocessing, feature extraction, and model training, reducing execution time. The dataset consists of MRI images classified as "yes" (tumor detected) and "no" (no tumor). Images undergo preprocessing, and features are extracted using Gray Level Co-occurrence Matrix (GLCM) properties for improved classification accuracy.

## Download & Run

1. **Download the ZIP** of this repository from GitHub.
2. **Unzip** it locally, the `data/brain_tumor_dataset` folder is already included.
3. Open the project folder in your editor or terminal.

## Project Structure

**Main Script**  
`main.py`: Entry point to run the entire brain tumor detection pipeline, from preprocessing to training and evaluation.

**Notebook**  
`notebooks/ImageProcessingForTumorDetection.ipynb`: Demonstrates the pipeline step-by-step for exploration, visualization, and experimentation.

**Source Code**  
`src/`: Contains all Python modules for the pipeline:  
- `data_loader.py`: Loads MRI images and labels from the dataset.  
- `preprocessing.py`: Applies image preprocessing (resize, normalization, noise reduction).  
- `sequential_filtering.py`: Sequential image filtering.  
- `parallel_filtering.py`: Parallelized filtering using multiprocessing.  
- `feature_extraction.py`: Extracts GLCM texture features from images.  
- `model.py`: Defines the machine learning model.  
- `train_model.py`: Handles model training and saving.  
- `evaluation.py`: Evaluates model performance (accuracy, confusion matrix, etc.).

**Dataset**  
`data/brain_tumor_dataset/`: MRI images labeled in two folders:  
- `yes/`: Tumor present  
- `no/`: Tumor absent 
