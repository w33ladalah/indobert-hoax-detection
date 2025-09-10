import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    """
    Class for preprocessing Indonesian text for hoax detection.
    Includes cleaning, tokenization, stopword removal, and stemming.
    """
    
    def __init__(self):
        # Initialize Indonesian stemmer
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        # Get Indonesian stopwords
        self.stop_words = set(stopwords.words('indonesian'))
        
        # Add custom stopwords if needed
        custom_stopwords = {"yg", "dgn", "nya", "utk"}
        self.stop_words.update(custom_stopwords)
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs, and normalizing whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove Indonesian stopwords from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """
        Apply Indonesian stemming to text.
        
        Args:
            text: Input text string
            
        Returns:
            Stemmed text
        """
        return self.stemmer.stem(text)
    
    def preprocess(self, text):
        """
        Apply full preprocessing pipeline: cleaning, stopword removal, and stemming.
        
        Args:
            text: Raw input text
            
        Returns:
            Fully preprocessed text
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text
