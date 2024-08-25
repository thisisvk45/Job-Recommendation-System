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
