import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfd = TfidfVectorizer(stop_words='engish')
nltk.download('punkt')
nltk.download('stopwords')

import os

# Replace with the path to your file
file_path = 'C:/Users/ACER/ml project/resume_scanning/tfidfd.pkl'
if os.path.exists(file_path):
    tfidfd = pickle.load(open(file_path, 'rb'))
else:
    print("File not found!")


#loading model
clf = pickle.load(open('clf.pkl', 'rb'))

 

 

def cleanResume(txt):
    # Remove URLs
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)
    
    # Remove RT or cc (used in retweets or replies)
    cleanTxt = re.sub(r'RT|cc', ' ', cleanTxt)
    
    # Remove @mentions
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    
    # Remove hashtags
    cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)
    
    # Remove punctuation and special characters
    cleanTxt = re.sub(r'[%s]' % re.escape(r"""|"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    
    # Remove non-ASCII characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    
    # Remove extra whitespace
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()

    return cleanTxt

#web app
def main():
    st.title('Resume Scanning')
uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

if uploaded_file is not None:
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')

    # Call the cleanResume function with the resume text (string, not list)
    cleaned_resume = cleanResume(resume_text)

# Use tfidf to transform the cleaned resume
    input_features = tfidfd.transform([cleaned_resume])

# Predict the label
    prediction_id = clf.predict(input_features)[0]

# Display the prediction result
    st.write(prediction_id)



#python main
if __name__== "__main__":
    main()

