
# Job Recommendation System

![Job Recommendation System](https://img.shields.io/badge/Streamlit-v1.14.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8-blue)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation of Key Components](#explanation-of-key-components)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project is a **Job Recommendation System** built using Streamlit, allowing users to upload their CVs (in PDF format) and receive tailored job recommendations based on the contents of their CV. The system processes the CV text, compares it with job descriptions, and uses several machine learning techniques to recommend the most relevant job openings.

## Features

- **PDF CV Upload:** Users can upload their CVs in PDF format.
- **Text Preprocessing:** The system tokenizes, removes stop words, and lemmatizes the text from both the CV and job descriptions.
- **Machine Learning Models:** Utilizes TF-IDF Vectorizer, CountVectorizer, and K-Nearest Neighbors (KNN) to compute similarity scores and recommend jobs.
- **Customizable Recommendations:** Users can select the number of job recommendations they wish to receive.
- **Interactive UI:** Built with Streamlit for an intuitive user experience.
- **User Feedback Loop:** Incorporates user feedback to refine and personalize future job recommendations.

## Requirements

- Python 3.8+
- Streamlit 1.14.0
- Pandas
- pdfplumber
- NLTK
- Scikit-learn
- NumPy

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/job-recommendation-system.git
   cd job-recommendation-system
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run candidate_app.py
   ```

## Usage

1. **Upload your CV:** Launch the app and upload your CV in PDF format.
2. **Receive Recommendations:** The app will process your CV and generate job recommendations.
3. **Provide Feedback:** Like or dislike the recommendations to help improve the accuracy of future recommendations.
4. **Customize Output:** Use the slider to adjust the number of job recommendations displayed.

## Explanation of Key Components

- **PDF Processing:** The system uses `pdfplumber` to extract text from PDF files.
- **Text Preprocessing:** NLTK is used for tokenization, stopword removal, and lemmatization.
- **TF-IDF and CountVectorizer:** These vectorizers are used to convert textual data into numerical form for similarity comparison.
- **K-Nearest Neighbors (KNN):** KNN is used to identify the most similar job descriptions based on the CV content.
- **Feedback Loop:** A simple feedback mechanism is implemented to collect user preferences and refine future recommendations.

## File Structure

- `candidate_app.py`: Main application file that runs the Streamlit app.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `job_recommendations.csv`: Sample dataset containing job descriptions and related information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have suggestions for improvements or find any bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the contributors and the open-source community for providing the tools and libraries that made this project possible.
```

This `README.md` file provides a comprehensive overview of your Job Recommendation System, including installation instructions, usage guidelines, and details about the key components. You can now include this file in your project to help others understand how to set up and use your system.
