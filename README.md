
# Job Recommendation System

![Job Recommendation System](https://img.shields.io/badge/Streamlit-v1.14.0-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8-blue)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation of Key Components](#explanation-of-key-components)
- [Data Availability](#data-availability)
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
   git clone https://github.com/thisisvk45/Job-Recommendation-System.git
   cd Job-Recommendation-System
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

To run the job recommendation system, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local server. Open your browser and navigate to `http://localhost:8501` to interact with the application.

## Explanation of Key Components

- **PDF Extraction:** The `pdfplumber` library is used to extract text from PDF files.
- **Text Preprocessing:** The text is tokenized, stop words are removed, and lemmatization is performed using NLTK.
- **Similarity Calculation:** TF-IDF Vectorizer, CountVectorizer, and KNN models are used to calculate similarities between the CV text and job descriptions.
- **Weighted Scoring:** Final recommendations are based on a weighted combination of similarity scores.

## Data Availability

The job data used for this project was scraped using Selenium. However, due to various reasons, I am not sharing the scraped data or the code used for web scraping. Instead, I have provided a sample dataset (`job_recommendations.csv`) that can be used for basic tasks and testing the application.

## File Structure

```
Job-Recommendation-System/
│
├── app.py                # Main application file
├── job_recommendations.csv  # Sample job data used for recommendations
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Files and directories to ignore in Git
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the intuitive framework.
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction.
- [NLTK](https://www.nltk.org/) for text preprocessing tools.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning libraries.
