import streamlit as st
import pandas as pd
import pdfplumber
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set Streamlit configuration
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(processed_tokens)

# Load job data
@st.cache_data
def load_job_data():
    return pd.read_csv('job_recommendations.csv')

# Load job data
job_data = load_job_data()

# Main layout
st.title('🔍 Job Recommendation System')
st.markdown("""
    Welcome to the Job Recommendation System! Upload your CV to receive tailored job recommendations.
    The system analyzes your CV and suggests job openings based on your experience and skills.
""")

# Sidebar options
with st.sidebar:
    st.header('Options')
    career_level = st.selectbox('Select Career Level:', ['Entry-Level', 'Mid-Level', 'Senior'])
    uploaded_cv = st.file_uploader('Upload your CV (PDF only)', type='pdf')
    num_recommendations = st.slider('Select the number of recommendations:', min_value=1, max_value=10, value=5, step=1)

# Main content area
if uploaded_cv:
    cv_text = ''
    with st.spinner('Extracting text from CV...'):
        cv_text = extract_text_from_pdf(uploaded_cv)
    st.success('Text extraction completed.')

    # Display extracted CV text
    st.subheader("Extracted Text from CV:")
    st.write(cv_text)

    # Process the CV text
    processed_cv_text = preprocess_text(cv_text)
    st.subheader("Processed CV Text:")
    st.write(processed_cv_text)

    if career_level:
        # Filter job data based on career level
        filtered_job_data = job_data[job_data['Level'] == career_level]

        if not filtered_job_data.empty:
            # Preprocess job descriptions
            filtered_job_data['processed_text'] = filtered_job_data['Description'].apply(preprocess_text)

            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_job_data['processed_text'])
            user_tfidf = tfidf_vectorizer.transform([processed_cv_text])
            tfidf_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

            # Combine and display results
            filtered_job_data['similarity'] = tfidf_similarities
            filtered_job_data = filtered_job_data.sort_values(by='similarity', ascending=False)
            top_jobs = filtered_job_data.head(num_recommendations)

            st.subheader("Top Job Recommendations:")
            for index, row in top_jobs.iterrows():
                st.markdown(f"### {row['Job Opening Title']} at {row['Company Name']}")
                st.write(f"**Location**: {row['Job Location']}")
                st.write(f"**Skills Needed**: {row['Skills Needed']}")
                st.write(f"**Salary**: ₹{row['Estimated Salary (INR)']}")
                st.write(f"**Description**: {row['Description']}")
                st.write(f"**Hiring Manager**: {row['Hiring Manager']}")
                st.write(f"**Hiring Manager Email**: {row['Contact Information']}")
                st.write(f"**Application Deadline**: {row['Application Deadline']}")
                st.progress(row['similarity'])

            # Visualization: Company Size Distribution
            st.subheader("Company Size Distribution of Recommended Jobs")
            size_count = top_jobs['Company Size'].value_counts()
            fig = px.pie(size_count, values=size_count.values, names=size_count.index, title='Company Size Distribution')
            st.plotly_chart(fig)

            # Visualization: Salary Range Distribution
            st.subheader("Salary Range Distribution")
            salary_range = top_jobs['Estimated Salary (INR)'].astype(float)
            fig = px.box(salary_range, y=salary_range, title='Salary Range Distribution')
            st.plotly_chart(fig)

            # Job Location Map
            st.subheader("Job Locations")
            geolocator = Nominatim(user_agent="job_recommender")
            locations = top_jobs['Job Location'].apply(lambda loc: geolocator.geocode(loc))
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center on India
            for loc in locations.dropna():
                folium.Marker([loc.latitude, loc.longitude], popup=loc.address).add_to(m)
            folium_static(m)

        else:
            st.write("No jobs found for the selected career level.")
else:
    st.sidebar.write("Please upload your CV to get recommendations.")

# Add footer
st.markdown("""
    ---
    **Disclaimer:** This system provides recommendations based on the analysis of your CV. 
    The job listings are simulated and may not correspond to real-world job openings.
""")

