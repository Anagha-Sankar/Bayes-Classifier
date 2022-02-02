# BAYE'S CLASSIFIER

### CONTENTS:

- Description

- Requirement

- Installation

- Directory Structure

- Usage

- Credit

  #### Description

  Bayesian Classifier uses the Bayes Theorem. Bayes Classifier is one of the simple and most effective Classification algorithms. It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.

  #### Requirement

  1. Python 3
  2. Jupyter Notebook
  3. Numpy
  4. Opencv
  5. PIL

  #### Installation

  1. **Python**

     - Visit and download Python from https://www.python.org/downloads/ 

     - Install and add Python to path

       ```
       python3 -V
       ```

       Install pip

       ```
       sudo apt install python3-pip
       ```

  2. **Numpy**

     Using pip,

     ```
     pip install numpy
     ```

  3. **Opencv**

     In command line, change directory to where pip is present

     ```
     pip install opencv-python
     ```

  4. **PIL**

     In command line or powershell

     ```
     python3 -m pip install --upgrade Pillow
     ```

  5. **Jupyter notebook**

     Install the classic Jupyter Notebook using:

     ```
     pip install notebook
     ```

     To run the notebook

     ```
     jupyter notebook
     ```

  #### Directory Structure

  ```
  .
  ├── src                     	# Source files
  │   ├── Bayes Classifier.ipynb	# Jupyter Notebook
  |	├──band1.jpg	# Input image 1
  |	├──band2.jpg	# Input image 2
  |	├──band3.jpg	# Input image 3
  |	├──band4.jpg	# Input image 4
  |	├── River		# 50 Coordinates of River class
  |	├── NonRiver	# 100 Coordinates of Non-River class
  |	├── Output_Images
  |		├── river03non07.jpeg
  |		├── river03non07.jpeg
  |		├── river03non07.jpeg
  ├── REPORT.md
  └── README.md
  ```
  
  #### Usage

  The source code (jupyter notebook) is present in the 'src' folder. The 4 input band images(R,G,B,I) are band1.jpg, band2.jpg, band3.jpg and band4.jpg. Coordinates of river and non-river for training pixel are in the files 'River' and 'NonRiver'. In the 'Output_Images' folder, the three output images for different Prior Probabilities are included.

  Project.md explains the source code.

 #### Credit
 To record coordinates,
 https://www.mobilefish.com/services/record_mouse_coordinates/record_mouse_coordinates.php
