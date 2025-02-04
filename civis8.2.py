import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Load BART for text generation (summarization)
generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Initialize Sentence Transformer for semantic search (retrieving relevant documents)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set up a simple list of trusted sources
trusted_sources = [
    "https://www.politifact.com/",
    "https://www.snopes.com/",
    "https://www.factcheck.org/"
]

# Function to extract content, publication date, and author from the URL
def extract_content_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article text
        article_text = " ".join([p.get_text() for p in soup.find_all('p')])

        # Extract publication date
        publication_date = extract_publication_date(soup)

        # Extract author
        author = extract_author(soup)

        return article_text, publication_date, author
    except Exception as e:
        return None, None, None

# Function to extract publication date
def extract_publication_date(soup):
    meta_date = soup.find('meta', attrs={'name': 'date'})
    if meta_date:
        return meta_date.get('content')
    
    meta_article_date = soup.find('meta', attrs={'property': 'article:published_time'})
    if meta_article_date:
        return meta_article_date.get('content')
    
    date_text = soup.find(text=re.compile(r"published|released", re.IGNORECASE))
    if date_text:
        return date_text.strip()
    
    return None

# Function to extract author
def extract_author(soup):
    meta_author = soup.find('meta', attrs={'name': 'author'})
    if meta_author:
        return meta_author.get('content')
    
    author_tag = soup.find('span', class_='author')
    if author_tag:
        return author_tag.get_text()
    
    return None

# Function for semantic search using Sentence Transformers
def retrieve_relevant_sources(claim, trusted_sources):
    claim_embedding = model.encode(claim, convert_to_tensor=True)
    
    relevant_sources = []
    
    for source_url in trusted_sources:
        try:
            response = requests.get(source_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            source_text = " ".join([p.get_text() for p in soup.find_all('p')])
            source_embedding = model.encode(source_text, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(claim_embedding, source_embedding)
            
            if similarity.item() > 0.5:  # Threshold for similarity
                relevant_sources.append(source_url)
        except Exception as e:
            print(f"Error processing {source_url}: {e}")
    
    return relevant_sources

# Function to detect sentiment and bias in text using VADER and TextBlob
def detect_bias_and_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vader_score = analyzer.polarity_scores(text)
    sentiment = "Neutral"
    
    if vader_score['compound'] >= 0.05:
        sentiment = "Positive"
    elif vader_score['compound'] <= -0.05:
        sentiment = "Negative"
    
    # Use TextBlob for polarity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    return sentiment, polarity

# Main function to test the claim
def combined_test(url):
    content, publication_date, author = extract_content_from_url(url)
    
    if not content:
        return "Error: Unable to extract content from the URL."
    
    # Extract the main claim (simplified version)
    claim = content.split(".")[0]
    
    # Retrieve relevant sources for the claim using semantic search
    relevant_sources = retrieve_relevant_sources(claim, trusted_sources)
    
    # Sentiment and bias analysis
    sentiment, polarity = detect_bias_and_sentiment(content)
    
    # Summary of results
    summary = f"Evaluation Summary for {url}:\n"
    summary += f"Sentiment: {sentiment}\n"
    summary += f"Polarity: {polarity}\n"
    summary += f"Publication Date: {publication_date}\n"
    summary += f"Author: {author}\n"
    
    if relevant_sources:
        summary += f"The following trusted sources support this claim:\n"
        for source in relevant_sources:
            summary += f"- {source}\n"
    else:
        summary += "No relevant sources were found supporting this claim.\n"
    
    return summary

# Example usage (just provide the URL)
url = "https://secretlifeofmom.com/grace-kelly-granddaughter-look-alike/?axqr=gv3pzh"
result = combined_test(url)
print(result)
