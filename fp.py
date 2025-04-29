# Flask Web Application with improved summarizer integration
# A machine learning application that identifies and highlights key sentences in text

import numpy as np
import re
import argparse
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import warnings
from flask import Flask, request, jsonify, render_template_string
import traceback

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")


# Initialize NLTK resources
def initialize_nltk():
    """Download required NLTK data"""
    print("Downloading NLTK resources...")
    try:
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        print("NLTK resources downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")
        print("Attempting alternate download method...")
        import subprocess
        try:
            # Try using system-level Python to download
            subprocess.call([sys.executable, '-m', 'nltk.downloader', 'punkt', 'stopwords'])
            print("NLTK resources downloaded using alternate method!")
        except Exception as e2:
            print(f"Failed to download NLTK resources: {str(e2)}")
            print("You may need to manually download NLTK resources using:")
            print("import nltk; nltk.download('punkt'); nltk.download('stopwords')")


# Ensure sample file exists
def ensure_sample_file_exists(filename="sample_article.txt"):
    """Check if sample file exists and create it if not"""
    if not os.path.exists(filename):
        print(f"Sample file '{filename}' not found. Creating sample file...")
        sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
        The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
        This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
        AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
        As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
        Machine learning is a subfield of AI that focuses on the development of algorithms that can access data and use it to learn for themselves.
        Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to analyze various factors of data.
        The field of AI research was founded at a workshop held on the campus of Dartmouth College in the summer of 1956.
        Those who attended would become the leaders of AI research for decades, and many of their students and colleagues made major contributions to the field.
        """
        
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(sample_text)
            print(f"Sample file '{filename}' created successfully!")
            return True
        except Exception as e:
            print(f"Error creating sample file: {str(e)}")
            return False
    return True


# Sample text for demonstration
def get_sample_text():
    """Return a sample text for demonstration"""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
    Machine learning is a subfield of AI that focuses on the development of algorithms that can access data and use it to learn for themselves.
    Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to analyze various factors of data.
    The field of AI research was founded at a workshop held on the campus of Dartmouth College in the summer of 1956.
    Those who attended would become the leaders of AI research for decades, and many of their students and colleagues made major contributions to the field.
    """


# Enhanced TextRank Summarizer with better preprocessing and keyword emphasis
class TextRankSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n', ' ', text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences but be less restrictive
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    def identify_key_terms(self, text, top_n=10):
        """Identify the most important terms in the text"""
        # Tokenize and filter out stop words
        words = [w.lower() for w in text.split() if w.lower() not in self.stop_words and len(w) > 3]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N words
        return [word for word, count in sorted_words[:top_n]]
    
    def create_sentence_vectors(self, sentences):
        # Create more effective vectors with better TF-IDF weighting
        # First, build a comprehensive word frequency dictionary
        word_freq = {}
        sentence_words = []
        
        for sentence in sentences:
            words = [w.lower() for w in sentence.split() if w.lower() not in self.stop_words]
            sentence_words.append(words)
            
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create vectors with improved TF-IDF weighting
        vectors = []
        for words in sentence_words:
            if not words:
                vectors.append(np.zeros((1,)))  # Empty vector for empty sentences
                continue
                
            # Count word frequencies in this sentence
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Create the vector with TF-IDF weighting
            vec = np.zeros((len(words),))
            for i, word in enumerate(words):
                # Term frequency in this sentence (normalized)
                tf = word_counts[word] / len(words)
                # Inverse document frequency with smoothing
                idf = np.log(len(sentences) / (word_freq[word] + 1))
                # TF-IDF with sublinear scaling
                vec[i] = (1 + np.log(tf)) * idf  # Sublinear TF scaling
            
            vectors.append(vec)
        
        return vectors
    
    def build_similarity_matrix(self, vectors):
        # Create similarity matrix
        sim_mat = np.zeros([len(vectors), len(vectors)])
        
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if i != j and vectors[i].size > 0 and vectors[j].size > 0:
                    # Use cosine similarity
                    sim_mat[i][j] = self.cosine_similarity_manual(vectors[i], vectors[j])
        
        return sim_mat
    
    def cosine_similarity_manual(self, vec1, vec2):
        # Ensure vectors are of same length for comparison by padding the shorter one
        if len(vec1) != len(vec2):
            if len(vec1) < len(vec2):
                # Pad vec1 with zeros
                vec1 = np.pad(vec1, (0, len(vec2) - len(vec1)), 'constant')
            else:
                # Pad vec2 with zeros
                vec2 = np.pad(vec2, (0, len(vec1) - len(vec2)), 'constant')
        
        if np.sum(vec1) == 0 or np.sum(vec2) == 0:
            return 0
        
        return np.dot(vec1, vec2) / (np.sqrt(np.sum(vec1**2)) * np.sqrt(np.sum(vec2**2)))
    
    def boost_key_sentences(self, sentences, sentence_scores, key_terms):
        """Boost scores of sentences containing key terms"""
        boosted_scores = []
        
        for i, (sentence, score) in enumerate(sentence_scores):
            sentence_lower = sentence.lower()
            boost = 0
            for term in key_terms:
                if term in sentence_lower:
                    boost += 0.1  # Add boost for each key term
            
            boosted_scores.append((sentence, score + boost))
        
        return boosted_scores
    
    def add_positional_weighting(self, sentences, sentence_scores):
        """Add weight to sentences based on their position"""
        position_weighted = sentence_scores.copy()
        
        # Boost first sentence
        if len(sentences) > 0:
            position_weighted[0] = (sentences[0], position_weighted[0][1] + 0.15)
        
        # Boost last sentence
        if len(sentences) > 1:
            position_weighted[-1] = (sentences[-1], position_weighted[-1][1] + 0.1)
        
        # Identify paragraph breaks and boost first sentences of paragraphs
        for i in range(1, len(sentences)):
            # Check if this might be a paragraph start
            if len(sentences[i-1].split()) < 10 or sentences[i-1].endswith(('.', '!', '?')):
                position_weighted[i] = (sentences[i], position_weighted[i][1] + 0.1)
        
        return position_weighted
    
    def get_sentence_scores(self, text, sentences):
        # Preprocess and get sentence vectors
        clean_sentences = sentences
        sentence_vectors = self.create_sentence_vectors(clean_sentences)
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentence_vectors)
        
        # Apply PageRank algorithm with more iterations for better convergence
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=200, tol=1e-06)
        
        # Create dictionary with sentence text and its score
        sentence_scores = [(clean_sentences[i], scores[i]) for i in range(len(clean_sentences))]
        
        # Identify key terms in the text
        key_terms = self.identify_key_terms(text)
        
        # Apply boosting for sentences containing key terms
        sentence_scores = self.boost_key_sentences(clean_sentences, sentence_scores, key_terms)
        
        # Apply positional weighting
        sentence_scores = self.add_positional_weighting(clean_sentences, sentence_scores)
        
        return sentence_scores, clean_sentences
    
    def adaptive_ratio(self, sentences):
        """Calculate an adaptive ratio based on text length"""
        sentence_count = len(sentences)
        
        # For shorter texts, highlight more
        if sentence_count < 5:
            return 0.6  # Highlight 60%
        elif sentence_count < 10:
            return 0.5  # Highlight 50%
        elif sentence_count < 20:
            return 0.4  # Highlight 40%
        else:
            return 0.35  # Default to 35% for longer texts
    
    def summarize(self, text, ratio=0.3):
        """
        Summarize the text using improved TextRank algorithm
        
        :param text: Input text to summarize
        :param ratio: Proportion of sentences to include in summary (0.0 to 1.0)
        :return: List of (sentence, score) tuples sorted by position in original text
        """
        # Preprocess text
        clean_sentences = self.preprocess_text(text)
        
        if not clean_sentences:
            return []
        
        # If adaptive ratio is requested (ratio=0), calculate based on text length
        if ratio <= 0:
            ratio = self.adaptive_ratio(clean_sentences)
            
        # Get sentence scores with all enhancements
        sentence_scores, _ = self.get_sentence_scores(text, clean_sentences)
        
        # Sort sentences by score in descending order
        ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on ratio, but use a minimum of 1 sentence
        # or 30% of sentences, whichever is greater
        min_sentences = max(1, int(len(clean_sentences) * 0.3))
        num_sentences = max(min_sentences, int(len(clean_sentences) * ratio))
        
        # Cap at 70% of sentences to avoid highlighting almost everything
        max_sentences = int(len(clean_sentences) * 0.7)
        num_sentences = min(num_sentences, max_sentences)
        
        top_sentences = ranked_sentences[:num_sentences]
        
        # Sort by position in original text
        original_order = []
        for sentence, score in top_sentences:
            index = clean_sentences.index(sentence)
            original_order.append((index, sentence, score))
        
        original_order.sort()
        
        # Return sentences with their scores
        return [(sentence, score) for _, sentence, score in original_order]


# Enhanced TF-IDF Summarizer with domain adaptation and key term emphasis
class TFIDFSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n', ' ', text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences but be less restrictive
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    def identify_key_terms(self, text, top_n=10):
        """Identify the most important terms in the text"""
        # Tokenize and filter out stop words
        words = [w.lower() for w in text.split() if w.lower() not in self.stop_words and len(w) > 3]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N words
        return [word for word, count in sorted_words[:top_n]]
    
    def detect_domain(self, text):
        """Detect the domain of the text based on keyword presence"""
        text_lower = text.lower()
        
        # Define domain-specific keywords
        domains = {
            'science': ['research', 'study', 'data', 'analysis', 'results', 'experiment', 
                      'scientific', 'hypothesis', 'methodology', 'findings'],
            'technology': ['software', 'hardware', 'technology', 'system', 'computer', 
                         'digital', 'internet', 'device', 'application', 'algorithm'],
            'business': ['business', 'company', 'market', 'customer', 'product', 
                      'service', 'revenue', 'sales', 'strategy', 'investment'],
            'health': ['health', 'medical', 'patient', 'disease', 'treatment', 
                    'hospital', 'doctor', 'clinical', 'therapy', 'diagnosis'],
            'education': ['education', 'learning', 'student', 'teacher', 'school', 
                       'academic', 'university', 'knowledge', 'classroom', 'curriculum']
        }
        
        # Count occurrences of domain-specific words
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return the domain with the highest score
        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def add_domain_weighting(self, sentences, sentence_scores, domain="general"):
        """Add domain-specific weighting to sentences"""
        boosted_scores = sentence_scores.copy()
        
        # Define domain-specific keywords
        domain_keywords = {
            'science': ["research", "study", "found", "results", "conclusion", 
                      "experiment", "data", "analysis", "evidence", "significant"],
            'technology': ["developed", "designed", "system", "technology", "platform",
                         "solution", "innovation", "implementation", "software", "hardware"],
            'business': ["market", "company", "business", "industry", "organization",
                       "customer", "profit", "strategy", "product", "service"],
            'health': ["health", "medical", "patient", "treatment", "clinical",
                     "diagnosis", "therapy", "symptoms", "condition", "disease"],
            'education': ["learning", "students", "teaching", "education", "school", 
                        "study", "knowledge", "skills", "understanding", "curriculum"]
        }
        
        # Use general keywords if domain not specified or recognized
        if domain not in domain_keywords:
            domain = "general"
            domain_keywords["general"] = ["important", "significant", "key", "main", 
                                       "essential", "critical", "fundamental", "notably"]
        
        # Boost sentences containing domain-specific terms
        for i, (sentence, score) in enumerate(sentence_scores):
            sentence_lower = sentence.lower()
            boost = 0
            
            for keyword in domain_keywords[domain]:
                if keyword in sentence_lower:
                    boost += 0.05  # Boost for each domain keyword
            
            boosted_scores[i] = (sentence, score + boost)
        
        return boosted_scores
    
    def add_positional_weighting(self, sentences, sentence_scores):
        """Add positional weighting to sentences"""
        position_weighted = sentence_scores.copy()
        
        # Boost first sentence
        if len(sentences) > 0:
            position_weighted[0] = (sentences[0], position_weighted[0][1] + 0.15)
        
        # Boost last sentence
        if len(sentences) > 1:
            position_weighted[-1] = (sentences[-1], position_weighted[-1][1] + 0.1)
        
        # Identify paragraph breaks and boost first sentences of paragraphs
        for i in range(1, len(sentences)):
            # Check if this might be a paragraph start
            if len(sentences[i-1].split()) < 10 or sentences[i-1].endswith(('.', '!', '?')):
                position_weighted[i] = (sentences[i], position_weighted[i][1] + 0.1)
        
        return position_weighted
    
    def adaptive_ratio(self, sentences):
        """Calculate an adaptive ratio based on text length"""
        sentence_count = len(sentences)
        
        # For shorter texts, highlight more
        if sentence_count < 5:
            return 0.6  # Highlight 60%
        elif sentence_count < 10:
            return 0.5  # Highlight 50%
        elif sentence_count < 20:
            return 0.4  # Highlight 40%
        else:
            return 0.35  # Default to 35% for longer texts
    
    def summarize(self, text, ratio=0.3):
        """
        Summarize text using improved TF-IDF scoring
        
        :param text: Input text to summarize
        :param ratio: Proportion of sentences to include in summary (0.0 to 1.0)
        :return: List of (sentence, score) tuples sorted by position in original text
        """
        # Preprocess text and get sentences
        sentences = self.preprocess_text(text)
        
        if not sentences:
            return []
            
        if len(sentences) <= 1:
            return [(sentences[0], 1.0)] if sentences else []
        
        # If adaptive ratio is requested (ratio=0), calculate based on text length
        if ratio <= 0:
            ratio = self.adaptive_ratio(sentences)
        
        # Create enhanced TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            norm='l2',
            smooth_idf=True,
            sublinear_tf=True,  # Apply sublinear tf scaling
            max_features=1000,  # Limit to most frequent words
            min_df=1            # Include words that appear at least once
        )
        
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate sentence scores based on sum of TF-IDF values
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            # Get the TF-IDF values for this sentence
            tfidf_values = tfidf_matrix[i].toarray()[0]
            
            # Calculate score - use mean of top X% of TF-IDF values
            # This focuses on the most important words rather than all words
            if len(tfidf_values) > 0:
                sorted_values = sorted(tfidf_values, reverse=True)
                top_n = max(1, int(len(sorted_values) * 0.3))  # Top 30% of values
                score = np.mean(sorted_values[:top_n]) * np.sum(tfidf_values > 0)  # Also factor in number of important words
            else:
                score = 0
                
            sentence_scores.append((sentence, score))
        
        # Identify key terms
        key_terms = self.identify_key_terms(text)
        
        # Boost sentences containing key terms
        for i, (sentence, score) in enumerate(sentence_scores):
            sentence_lower = sentence.lower()
            boost = 0
            
            for term in key_terms:
                if term in sentence_lower:
                    boost += 0.1
            
            sentence_scores[i] = (sentence, score + boost)
        
        # Detect domain and add domain-specific weighting
        domain = self.detect_domain(text)
        sentence_scores = self.add_domain_weighting(sentences, sentence_scores, domain)
        
        # Add positional weighting
        sentence_scores = self.add_positional_weighting(sentences, sentence_scores)
        
        # Sort sentences by score in descending order
        ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # Select top sentences based on ratio, with smart minimum and maximum
        min_sentences = max(1, int(len(sentences) * 0.3))
        num_sentences = max(min_sentences, int(len(sentences) * ratio))
        max_sentences = int(len(sentences) * 0.7)
        num_sentences = min(num_sentences, max_sentences)
        
        top_sentences = ranked_sentences[:num_sentences]
        
        # Sort by position in original text to maintain original flow
        original_order = []
        for sentence, score in top_sentences:
            index = sentences.index(sentence)
            original_order.append((index, sentence, score))
        
        original_order.sort()
        
        # Return sentences with their scores
        return [(sentence, score) for _, sentence, score in original_order]


# Improved HTML Template with enhanced JS for better highlighting
HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retro-Scan</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0a0a0a;
            color: #fff;
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }
        .stars-container {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        .container {
            position: relative;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            z-index: 1;
        }
        h1 {
            color: #b700ff;
            text-align: center;
            font-size: 2.5rem;
            text-shadow: 0 0 10px rgba(183, 0, 255, 0.7);
        }
        .subtitle {
            text-align: center;
            color: #9c5bba;
            margin-top: -10px;
            margin-bottom: 40px;
        }
        .card {
            background-color: rgba(20, 20, 30, 0.8);
            border-radius: 10px;
            border: 1px solid #3d1363;
            padding: 25px;
            box-shadow: 0 0 20px rgba(183, 0, 255, 0.2);
            backdrop-filter: blur(5px);
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            justify-content: center;
            margin-bottom: 20px;
            padding: 15px;
            background-color: rgba(30, 20, 40, 0.7);
            border-radius: 8px;
            border: 1px solid #3d1363;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 150px;
        }
        label {
            font-weight: 500;
            color: #b18bd0;
        }
        select, input:not([type="range"]) {
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #3d1363;
            background-color: #1e1429;
            color: white;
            font-size: 0.9rem;
        }
        input[type="range"] {
            height: 5px;
            -webkit-appearance: none;
            width: 100%;
            background: #1e1429;
            border-radius: 5px;
            background-image: linear-gradient(to right, #3d1363, #b700ff);
            background-size: 30% 100%;
            background-repeat: no-repeat;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 15px;
            width: 15px;
            border-radius: 50%;
            background: #b700ff;
            cursor: pointer;
            box-shadow: 0 0 10px #b700ff;
        }
        input[type="color"] {
            -webkit-appearance: none;
            border: none;
            width: 80px;
            height: 30px;
            cursor: pointer;
            background-color: transparent;
        }
        input[type="color"]::-webkit-color-swatch-wrapper {
            padding: 0;
        }
        input[type="color"]::-webkit-color-swatch {
            border: 1px solid #3d1363;
            border-radius: 4px;
        }
        .buttons-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 16px;
            background-color: #b700ff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
            font-size: 0.9rem;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
            box-shadow: 0 0 10px rgba(183, 0, 255, 0.3);
        }
        button:hover {
            background-color: #9900d5;
            box-shadow: 0 0 15px rgba(183, 0, 255, 0.5);
        }
        textarea {
            width: 1120px;
            height: 250px;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #3d1363;
            background-color: #1e1429;
            color: white;
            resize: vertical;
            font-size: 1rem;
            line-height: 1.5;
            font-family: inherit;
        }
        .output-container {
            position: relative;
        }
        .output {
            border: 1px solid #3d1363;
            border-radius: 4px;
            padding: 25px;
            min-height: 250px;
            background-color: rgba(30, 20, 41, 0.7);
            line-height: 1.6;
            overflow-y: auto;
            max-height: 500px;
        }
        .highlighted {
            background-color: #b700ff;
            color: white;
            font-weight: 500;
            padding: 2px 5px;
            border-radius: 2px;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }
        .user-highlighted {
            background-color: #4d7cff;
            color: white;
            font-weight: 500;
            padding: 2px 5px;
            border-radius: 2px;
        }
        .loading {
            text-align: center;
            color: #b18bd0;
            padding: 20px;
        }
        .error {
            color: #ff5277;
            text-align: center;
            padding: 10px;
            background-color: rgba(255, 82, 119, 0.1);
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid rgba(255, 82, 119, 0.3);
        }
        .stats {
            margin-top: 20px;
            font-size: 0.9rem;
            color: #9c5bba;
            background-color: rgba(30, 20, 40, 0.7);
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #3d1363;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        .stat-box {
            background-color: rgba(20, 20, 30, 0.8);
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            border: 1px solid #3d1363;
        }
        .stat-value {
            font-size: 1.2rem;
            font-weight: 500;
            color: #b700ff;
            margin-top: 5px;
            text-shadow: 0 0 5px rgba(183, 0, 255, 0.3);
        }
        .copy-button {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(30, 20, 40, 0.8);
            border: 1px solid #3d1363;
            padding: 5px 10px;
            border-radius: 4px;
            color: #b18bd0;
            font-size: 0.8rem;
            cursor: pointer;
        }
        .copy-button:hover {
            background-color: rgba(61, 19, 99, 0.8);
            color: white;
        }
        .highlight-controls {
            margin-top: 10px;
            padding: 0 10px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .highlight-toggle {
            color: #b18bd0;
            text-decoration: underline;
            cursor: pointer;
            font-size: 0.9rem;
            transition: color 0.3s;
        }
        .highlight-toggle:hover {
            color: #b700ff;
            text-shadow: 0 0 5px rgba(183, 0, 255, 0.3);
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(183, 0, 255, 0.3);
            border-radius: 50%;
            border-top-color: #b700ff;
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .key-terms {
            margin-top: 15px;
            padding: 10px;
            background-color: rgba(30, 20, 40, 0.7);
            border-radius: 4px;
            font-size: 0.9rem;
            border: 1px solid #3d1363;
        }
        .key-terms-title {
            font-weight: 500;
            margin-bottom: 5px;
            color: #b18bd0;
        }
        .key-term {
            display: inline-block;
            margin: 3px;
            padding: 2px 8px;
            background-color: rgba(183, 0, 255, 0.2);
            border-radius: 10px;
            font-size: 0.85rem;
            border: 1px solid rgba(183, 0, 255, 0.3);
            color: #d0b8ff;
        }
        .click-instruction {
            font-style: italic;
            margin-top: 10px;
            color: #9c5bba;
            font-size: 0.85rem;
            text-align: center;
        }
        
        /* Floating background elements */
        .floating-shape {
            position: fixed;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, rgba(183, 0, 255, 0.8), rgba(183, 0, 255, 0));
            filter: blur(30px);
            opacity: 0.2;
            pointer-events: none;
            z-index: -1;
        }
        #shape1 {
            width: 400px;
            height: 400px;
            top: -100px;
            left: -100px;
            animation: floatShape1 20s ease-in-out infinite;
        }
        #shape2 {
            width: 500px;
            height: 500px;
            bottom: -150px;
            right: -150px;
            background: radial-gradient(circle at 70% 70%, rgba(77, 124, 255, 0.8), rgba(77, 124, 255, 0));
            animation: floatShape2 25s ease-in-out infinite;
        }
        @keyframes floatShape1 {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(100px, 50px); }
        }
        @keyframes floatShape2 {
            0%, 100% { transform: translate(0, 0); }
            50% { transform: translate(-100px, -70px); }
        }
    </style>
</head>
<body>
    <!-- Canvas for background animation -->
    <canvas id="starsCanvas" class="stars-container"></canvas>
    
    <!-- Floating shapes -->
    <div id="shape1" class="floating-shape"></div>
    <div id="shape2" class="floating-shape"></div>

    <div class="container">
        <h1>RETRO-SCAN</h1>
        <p class="subtitle">Important lines highlighting for faster reading with complete understanding.</p>
        
        <div class="card">
            <div class="controls">
                <div class="control-group">
                    <label for="algorithm">Algorithm:</label>
                    <select id="algorithm">
                        <option value="textrank">TextRank</option>
                        <option value="tfidf">TF-IDF</option>
                        <option value="auto">Auto-detect</option>
                    </select>
                </div>
                <div class="control-group">
                    <label for="ratio">Highlight Percentage:</label>
                    <input type="range" id="ratio" min="0" max="70" value="30" step="5">
                    <span id="ratio-value">30%</span>
                </div>
                <div class="control-group">
                    <label for="highlight-color">Highlight Color:</label>
                    <input type="color" id="highlight-color" value="#b700ff">
                </div>
                <div class="buttons-container">
                    <button id="process-btn">Process Text</button>
                    <button id="sample-btn">Load Sample</button>
                    <button id="clear-btn">Clear All</button>
                </div>
            </div>
            
            <textarea id="input-text" placeholder="Paste your article or text here..."></textarea>
            
            <div class="output-container">
                <div id="output" class="output">
                    <p>Processed text will appear here with important sentences highlighted.</p>
                </div>
                <button id="copy-btn" class="copy-button" style="display:none;">Copy All</button>
                <div class="highlight-controls" style="display:none;">
                    <span id="highlight-only-toggle" class="highlight-toggle">Show highlighted sentences only</span>
                    <span id="highlight-feedback" class="highlight-toggle">Enable user highlighting</span>
                </div>
                <p id="click-instruction" class="click-instruction" style="display:none;">
                    Click on any non-highlighted sentence to mark it as important. Click again to unmark.
                </p>
            </div>
            
            <div id="key-terms" class="key-terms" style="display:none;">
                <div class="key-terms-title">Key Terms Identified:</div>
                <div id="key-terms-list"></div>
            </div>
            
            <div id="stats" class="stats">
                <strong>Statistics</strong>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div>Original Word Count</div>
                        <div id="original-count" class="stat-value">0</div>
                    </div>
                    <div class="stat-box">
                        <div>Highlighted Word Count</div>
                        <div id="highlighted-count" class="stat-value">0</div>
                    </div>
                    <div class="stat-box">
                        <div>Highlight Percentage</div>
                        <div id="highlight-percentage" class="stat-value">0%</div>
                    </div>
                    <div class="stat-box">
                        <div>Sentences Highlighted</div>
                        <div id="highlight-count" class="stat-value">0</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Background Animation
            const canvas = document.getElementById('starsCanvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size to window size
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            // Particle system
            const particles = [];
            const particleCount = 100;
            
            // Create particles
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    radius: Math.random() * 2 + 0.5,
                    color: `rgba(${Math.floor(Math.random() * 100 + 150)}, ${Math.floor(Math.random() * 50)}, ${Math.floor(Math.random() * 100 + 150)}, ${Math.random() * 0.5 + 0.5})`,
                    speedX: Math.random() * 0.2 - 0.1,
                    speedY: Math.random() * 0.2 - 0.1,
                    pulse: Math.random() * 0.1
                });
            }
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw and update particles
                particles.forEach(particle => {
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
                    ctx.fillStyle = particle.color;
                    ctx.fill();
                    
                    // Move particles
                    particle.x += particle.speedX;
                    particle.y += particle.speedY;
                    
                    // Pulse size
                    particle.radius += Math.sin(Date.now() * 0.001) * particle.pulse;
                    
                    // Wrap around edges
                    if (particle.x < 0) particle.x = canvas.width;
                    if (particle.x > canvas.width) particle.x = 0;
                    if (particle.y < 0) particle.y = canvas.height;
                    if (particle.y > canvas.height) particle.y = 0;
                });
            }
            
            // Start animation
            animate();
            
            // Handle window resize
            window.addEventListener('resize', function() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            });

            // *** SPEED READER FUNCTIONALITY ***

            const algorithmSelect = document.getElementById('algorithm');
            const ratioInput = document.getElementById('ratio');
            const ratioValue = document.getElementById('ratio-value');
            const processBtn = document.getElementById('process-btn');
            const sampleBtn = document.getElementById('sample-btn');
            const clearBtn = document.getElementById('clear-btn');
            const inputText = document.getElementById('input-text');
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            const copyBtn = document.getElementById('copy-btn');
            const highlightControls = document.querySelector('.highlight-controls');
            const highlightOnlyToggle = document.getElementById('highlight-only-toggle');
            const highlightFeedbackToggle = document.getElementById('highlight-feedback');
            const clickInstruction = document.getElementById('click-instruction');
            const highlightColorInput = document.getElementById('highlight-color');
            const keyTermsSection = document.getElementById('key-terms');
            const keyTermsList = document.getElementById('key-terms-list');
            
            let originalSentences = [];
            let highlightedSentences = [];
            let showHighlightedOnly = false;
            let userHighlightingEnabled = false;
            let userHighlightedSentences = [];
            
            // Update ratio display value
            ratioInput.addEventListener('input', function() {
                if (parseInt(ratioInput.value) === 0) {
                    ratioValue.textContent = "Auto";
                } else {
                    ratioValue.textContent = ratioInput.value + '%';
                }
            });
            
            // Sample button click handler
            sampleBtn.addEventListener('click', function() {
                inputText.value = `Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
Machine learning is a subfield of AI that focuses on the development of algorithms that can access data and use it to learn for themselves.
Deep learning, a subset of machine learning, uses artificial neural networks with multiple layers to analyze various factors of data.
The field of AI research was founded at a workshop held on the campus of Dartmouth College in the summer of 1956.
Those who attended would become the leaders of AI research for decades, and many of their students and colleagues made major contributions to the field.`;
            });
            
            // Clear button click handler
            clearBtn.addEventListener('click', function() {
                inputText.value = '';
                output.innerHTML = '<p>Processed text will appear here with important sentences highlighted.</p>';
                stats.style.display = 'block';
                copyBtn.style.display = 'none';
                highlightControls.style.display = 'none';
                clickInstruction.style.display = 'none';
                keyTermsSection.style.display = 'none';
                document.getElementById('original-count').textContent = '0';
                document.getElementById('highlighted-count').textContent = '0';
                document.getElementById('highlight-percentage').textContent = '0%';
                document.getElementById('highlight-count').textContent = '0';
                userHighlightedSentences = [];
            });
            
            // Copy button click handler
            copyBtn.addEventListener('click', function() {
                // Create a temporary textarea to copy the text
                const tempTextArea = document.createElement('textarea');
                tempTextArea.value = output.innerText;
                document.body.appendChild(tempTextArea);
                tempTextArea.select();
                document.execCommand('copy');
                document.body.removeChild(tempTextArea);
                
                // Change the button text temporarily
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 2000);
            });
            
            // Highlight color change handler
            highlightColorInput.addEventListener('input', function() {
                const highlightColor = highlightColorInput.value;
                
                // Update all highlighted spans
                const highlightedSpans = document.querySelectorAll('.highlighted');
                highlightedSpans.forEach(span => {
                    span.style.backgroundColor = highlightColor;
                });
                
                // Store the color preference in localStorage
                localStorage.setItem('highlightColor', highlightColor);
            });
            
            // Load saved highlight color from localStorage if available
            const savedColor = localStorage.getItem('highlightColor');
            if (savedColor) {
                highlightColorInput.value = savedColor;
            }
            
            // Toggle between showing all sentences or only highlighted ones
            highlightOnlyToggle.addEventListener('click', function() {
                showHighlightedOnly = !showHighlightedOnly;
                
                if (showHighlightedOnly) {
                    // Show only highlighted sentences
                    highlightOnlyToggle.textContent = 'Show all sentences';
                    displayHighlightedOnly();
                } else {
                    // Show all sentences
                    highlightOnlyToggle.textContent = 'Show highlighted sentences only';
                    displayAllSentences();
                }
            });
            
            // Toggle user feedback mode
            highlightFeedbackToggle.addEventListener('click', function() {
                userHighlightingEnabled = !userHighlightingEnabled;
                
                if (userHighlightingEnabled) {
                    highlightFeedbackToggle.textContent = 'Disable user highlighting';
                    clickInstruction.style.display = 'block';
                    setupSentenceClicking();
                } else {
                    highlightFeedbackToggle.textContent = 'Enable user highlighting';
                    clickInstruction.style.display = 'none';
                }
            });
            
            function setupSentenceClicking() {
                // Add click handlers to all non-highlighted sentences
                const sentences = document.querySelectorAll('.output > span:not(.highlighted)');
                sentences.forEach(span => {
                    span.style.cursor = 'pointer';
                    span.addEventListener('click', function() {
                        // Toggle user highlighting
                        if (this.classList.contains('user-highlighted')) {
                            this.classList.remove('user-highlighted');
                            
                            // Remove from user highlighted sentences
                            const index = userHighlightedSentences.indexOf(this.textContent);
                            if (index > -1) {
                                userHighlightedSentences.splice(index, 1);
                            }
                        } else {
                            this.classList.add('user-highlighted');
                            userHighlightedSentences.push(this.textContent);
                        }
                        
                        // Update stats
                        updateHighlightStats();
                    });
                });
            }
            
            function displayHighlightedOnly() {
                let htmlContent = '';
                
                // Combine algorithm-highlighted and user-highlighted sentences
                const allHighlighted = [...highlightedSentences, ...userHighlightedSentences];
                
                allHighlighted.forEach((sentence, index) => {
                    const isUserHighlighted = userHighlightedSentences.includes(sentence);
                    const className = isUserHighlighted ? 'user-highlighted' : 'highlighted';
                    const style = isUserHighlighted ? '' : `style="background-color: ${highlightColorInput.value};"`;
                    
                    htmlContent += `<span class="${className}" ${style}>${sentence}</span>`;
                    
                    // Add spacing between sentences
                    if (index < allHighlighted.length - 1) {
                        htmlContent += '<br><br>';
                    }
                });
                
                output.innerHTML = htmlContent || '<p>No highlighted sentences found.</p>';
            }
            
            function displayAllSentences() {
                displayHighlightedText(inputText.value, window.lastSummaryData);
            }
            
            function updateHighlightStats() {
                // Count words in highlighted and user-highlighted content
                const highlightedSpans = document.querySelectorAll('.highlighted');
                const userHighlightedSpans = document.querySelectorAll('.user-highlighted');
                
                const highlightedWordCount = Array.from(highlightedSpans).reduce((count, span) => 
                    count + span.textContent.split(/\s+/).filter(word => word.length > 0).length, 0);
                
                const userHighlightedWordCount = Array.from(userHighlightedSpans).reduce((count, span) => 
                    count + span.textContent.split(/\s+/).filter(word => word.length > 0).length, 0);
                
                const totalHighlightedWordCount = highlightedWordCount + userHighlightedWordCount;
                const originalWordCount = parseInt(document.getElementById('original-count').textContent);
                
                document.getElementById('highlighted-count').textContent = totalHighlightedWordCount;
                document.getElementById('highlight-percentage').textContent = 
                    Math.round(totalHighlightedWordCount/originalWordCount*100) + '%';
                document.getElementById('highlight-count').textContent = 
                    highlightedSpans.length + userHighlightedSpans.length;
            }
            
            // Process button click handler
            processBtn.addEventListener('click', async function() {
                const text = inputText.value.trim();
                
                if (!text) {
                    output.innerHTML = '<p class="error">Please enter some text to process.</p>';
                    stats.style.display = 'block';
                    copyBtn.style.display = 'none';
                    highlightControls.style.display = 'none';
                    clickInstruction.style.display = 'none';
                    keyTermsSection.style.display = 'none';
                    return;
                }
                
                // Show loading indicator
                output.innerHTML = '<div class="loading"><div class="spinner"></div> Processing text...</div>';
                stats.style.display = 'block';
                copyBtn.style.display = 'none';
                highlightControls.style.display = 'none';
                clickInstruction.style.display = 'none';
                keyTermsSection.style.display = 'none';
                
                try {
                    // Reset user highlighted sentences
                    userHighlightedSentences = [];
                    
                    const response = await fetch('/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            algorithm: algorithmSelect.value,
                            ratio: parseInt(ratioInput.value) / 100,
                            include_key_terms: true
                        })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`API error: ${errorText}`);
                    }
                    
                    const data = await response.json();
                    
                    if (!data.summary || data.summary.length === 0) {
                        output.innerHTML = '<p class="error">Could not generate summary. Please try with different text or settings.</p>';
                        stats.style.display = 'block';
                        return;
                    }
                    
                    // Save the summary data for toggle functionality
                    window.lastSummaryData = data.summary;
                    
                    // Display the highlighted text
                    displayHighlightedText(text, data.summary);
                    
                    // Show key terms if available
                    if (data.key_terms && data.key_terms.length > 0) {
                        displayKeyTerms(data.key_terms);
                    }
                    
                    // Show copy button and highlight controls
                    copyBtn.style.display = 'block';
                    highlightControls.style.display = 'flex';
                    stats.style.display = 'block';
                    
                    // Display stats
                    const originalWordCount = text.split(/\s+/).filter(word => word.length > 0).length;
                    const summaryWordCount = data.summary.reduce((count, item) => 
                        count + item.sentence.split(/\s+/).filter(word => word.length > 0).length, 0);
                    
                    document.getElementById('original-count').textContent = originalWordCount;
                    document.getElementById('highlighted-count').textContent = summaryWordCount;
                    document.getElementById('highlight-percentage').textContent = 
                        Math.round(summaryWordCount/originalWordCount*100) + '%';
                    document.getElementById('highlight-count').textContent = data.summary.length;
                    
                } catch (error) {
                    console.error('Error processing text:', error);
                    output.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    stats.style.display = 'block';
                    copyBtn.style.display = 'none';
                    highlightControls.style.display = 'none';
                    clickInstruction.style.display = 'none';
                }
            });
            
            function displayKeyTerms(keyTerms) {
                keyTermsList.innerHTML = '';
                
                keyTerms.forEach(term => {
                    const termSpan = document.createElement('span');
                    termSpan.className = 'key-term';
                    termSpan.textContent = term;
                    keyTermsList.appendChild(termSpan);
                });
                
                keyTermsSection.style.display = 'block';
            }
            
            // Improved sentence highlighting function
            function displayHighlightedText(text, summaryData) {
                // Get important sentences
                const importantSentences = summaryData.map(item => item.sentence);
                highlightedSentences = importantSentences; // Store for toggle functionality
                
                // We need a more sophisticated sentence splitter than simple regex
                // This handles various edge cases better
                function splitIntoSentences(text) {
                    // Handle abbreviations, decimal numbers, and other special cases
                    // Replace these patterns temporarily to avoid wrong splitting
                    const preparedText = text
                        .replace(/([A-Z][a-z]*\.)(?=[A-Z])/g, '$1@@@') // Handle abbreviations like U.S.A.
                        .replace(/(\d+\.\d+)/g, (m) => m.replace('.', '@@@')); // Handle decimal numbers
                    
                    // Split by standard sentence terminators
                    const roughSplits = preparedText.split(/([.!?]+[\s\n]+|[.!?]+$)/);
                    
                    // Recombine the split parts to form sentences
                    const sentences = [];
                    for (let i = 0; i < roughSplits.length; i += 2) {
                        if (i + 1 < roughSplits.length) {
                            sentences.push((roughSplits[i] + roughSplits[i+1]).replace(/@@@/g, '.'));
                        } else if (roughSplits[i]) {
                            sentences.push(roughSplits[i].replace(/@@@/g, '.'));
                        }
                    }
                    
                    // Filter out empty sentences
                    return sentences.filter(s => s.trim().length > 0);
                }
                
                // Get sentences using our improved splitter
                const sentences = splitIntoSentences(text);
                originalSentences = sentences; // Store for toggle functionality
                
                // Create a function to normalize text for better matching
                function normalizeText(text) {
                    return text.replace(/\s+/g, ' ').trim().toLowerCase();
                }
                
                // Normalize all important sentences for comparison
                const normalizedImportantSentences = importantSentences.map(s => normalizeText(s));
                
                // Enhanced sentence importance detection with improved fuzzy matching
                function isImportantSentence(sentence) {
                    const normalizedSentence = normalizeText(sentence);
                    
                    // Check for exact match first
                    if (normalizedImportantSentences.includes(normalizedSentence)) {
                        return true;
                    }
                    
                    // Check for substantial overlap with any important sentence
                    const sentenceWords = normalizedSentence.split(/\s+/);
                    
                    for (const importantSentence of normalizedImportantSentences) {
                        const importantWords = importantSentence.split(/\s+/);
                        
                        // Count matching words (only count substantial words)
                        let matchCount = 0;
                        for (const word of sentenceWords) {
                            if (importantWords.includes(word) && word.length > 3) {
                                matchCount++;
                            }
                        }
                        
                        // Calculate overlap percentage - use the shorter sentence for denominator
                        const overlapRatio = matchCount / Math.min(sentenceWords.length, importantWords.length);
                        
                        if (overlapRatio > 0.7) { // 70% overlap threshold
                            return true;
                        }
                        
                        // Check if this sentence contains an important sentence
                        if (normalizedSentence.includes(importantSentence) && 
                            importantSentence.split(/\s+/).length > 4) { // Only if important sentence is substantial
                            return true;
                        }
                        
                        // Check if an important sentence contains this sentence
                        if (importantSentence.includes(normalizedSentence) && 
                            normalizedSentence.split(/\s+/).length > 4) { // Only if this sentence is substantial
                            return true;
                        }
                        
                        // Check for semantic similarity using keyword overlap
                        const sentenceKeywords = sentenceWords.filter(word => word.length > 4);
                        const importantKeywords = importantWords.filter(word => word.length > 4);
                        
                        if (sentenceKeywords.length > 0 && importantKeywords.length > 0) {
                            const keywordMatches = sentenceKeywords.filter(word => importantKeywords.includes(word));
                            const keywordOverlap = keywordMatches.length / Math.min(sentenceKeywords.length, importantKeywords.length);
                            
                            if (keywordOverlap > 0.8) { // 80% keyword overlap
                                return true;
                            }
                        }
                    }
                    
                    // Additional check: Look for sentences with a high density of key terms
                    // This is handled by the server-side scoring, but we could add more here if needed
                    
                    return false;
                }
                
                // Build the HTML with highlighted sentences
                let htmlContent = '';
                let highlightCount = 0;
                const highlightColor = highlightColorInput.value;
                
                sentences.forEach(sentence => {
                    // Check if this sentence is either algorithm-highlighted or user-highlighted
                    const isHighlighted = isImportantSentence(sentence);
                    const isUserHighlighted = userHighlightedSentences.includes(sentence);
                    
                    if (isHighlighted) {
                        htmlContent += `<span class="highlighted" style="background-color: ${highlightColor};">${sentence}</span> `;
                        highlightCount++;
                    } else if (isUserHighlighted) {
                        htmlContent += `<span class="user-highlighted">${sentence}</span> `;
                    } else {
                        htmlContent += `<span>${sentence}</span> `;
                    }
                });
                
                // Show a message if no sentences were highlighted
                if (highlightCount === 0 && importantSentences.length > 0) {
                    output.innerHTML = htmlContent + '<p class="error">Warning: No sentences were highlighted. Try adjusting the algorithm or ratio.</p>';
                } else {
                    output.innerHTML = htmlContent || '<p>No sentences found to highlight.</p>';
                }
                
                // Add highlight count to stats
                document.getElementById('highlight-count').textContent = highlightCount;
                
                // Setup user highlighting if enabled
                if (userHighlightingEnabled) {
                    setupSentenceClicking();
                    clickInstruction.style.display = 'block';
                }
            }
            
            // Initialize range slider background
            updateRangeBackground();
            
            // Update range slider background on input
            ratioInput.addEventListener('input', function() {
                updateRangeBackground();
            });
            
            function updateRangeBackground() {
                const value = (ratioInput.value - ratioInput.min) / (ratioInput.max - ratioInput.min) * 100;
                ratioInput.style.backgroundSize = value + '% 100%';
            }
            
            // Add some pulse effects to neon elements
            setInterval(() => {
                document.querySelector('h1').style.textShadow = `0 0 ${5 + 3 * Math.sin(Date.now() * 0.002)}px rgba(183, 0, 255, 0.7)`;
                
                const buttons = document.querySelectorAll('button:not(.copy-button)');
                buttons.forEach(button => {
                    button.style.boxShadow = `0 0 ${8 + 5 * Math.sin(Date.now() * 0.003)}px rgba(183, 0, 255, 0.3)`;
                });
            }, 50);
        });
    </script>
</body>
</html>
'''


def start_web_app():
    app = Flask(__name__)
    
    # Initialize NLTK first
    print("Initializing NLTK resources for web app...")
    initialize_nltk()
    
    # Initialize summarizers
    print("Initializing summarizers...")
    try:
        text_rank = TextRankSummarizer()
        tfidf = TFIDFSummarizer()
        print("Summarizers initialized successfully!")
    except Exception as e:
        print(f"Error initializing summarizers: {str(e)}")
        traceback.print_exc()
        print("Will attempt to initialize when needed.")
        text_rank = None
        tfidf = None
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/summarize', methods=['POST'])
    def summarize():
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'No text provided'}), 400
                
            text = data.get('text')
            algorithm = data.get('algorithm', 'textrank')  # Default to TextRank
            ratio = float(data.get('ratio', 0.3))  # Default to 30% of sentences
            include_key_terms = data.get('include_key_terms', False)
            
            # Validate ratio
            if ratio < 0 or ratio > 1:
                # If ratio is 0, use adaptive ratio instead
                if ratio == 0:
                    ratio = -1  # Signal to use adaptive ratio
                else:
                    return jsonify({'error': 'Ratio must be between 0 and 1'}), 400
            
            # Initialize summarizers if not done already
            nonlocal text_rank, tfidf
            if text_rank is None:
                text_rank = TextRankSummarizer()
            if tfidf is None:
                tfidf = TFIDFSummarizer()
            
            # Auto-detect best algorithm if requested
            if algorithm.lower() == 'auto':
                # Detect text characteristics to choose algorithm
                sentences = text_rank.preprocess_text(text)
                
                # Use TextRank for longer, more complex texts
                # Use TF-IDF for shorter, more factual texts
                avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
                
                if len(sentences) > 15 or avg_sentence_length > 20:
                    algorithm = 'textrank'  # Better for longer, complex texts
                else:
                    algorithm = 'tfidf'  # Better for shorter, factual texts
                
                print(f"Auto-detected algorithm: {algorithm}")
                
            # Select algorithm
            if algorithm.lower() == 'textrank':
                result = text_rank.summarize(text, ratio)
                if include_key_terms:
                    key_terms = text_rank.identify_key_terms(text)
            elif algorithm.lower() == 'tfidf':
                result = tfidf.summarize(text, ratio)
                if include_key_terms:
                    key_terms = tfidf.identify_key_terms(text)
            else:
                return jsonify({'error': 'Invalid algorithm specified'}), 400
                
            # Convert to list of dicts for JSON serialization
            summary_data = [{"sentence": sentence, "score": float(score)} for sentence, score in result]
            
            response_data = {
                'summary': summary_data,
                'algorithm': algorithm,
                'ratio': ratio if ratio > 0 else 'adaptive'
            }
            
            # Include key terms if requested
            if include_key_terms:
                response_data['key_terms'] = key_terms
                
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    print("Starting web application. Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)


# Main function for command-line usage
def main():
    parser = argparse.ArgumentParser(description='AI Speed Reader Pro - Extract important sentences from text')
    parser.add_argument('--file', type=str, help='Path to the text file to summarize')
    parser.add_argument('--algorithm', type=str, default='textrank', 
                        choices=['textrank', 'tfidf', 'auto'], 
                        help='Summarization algorithm to use')
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='Proportion of sentences to include (0.0-1.0 or 0 for adaptive)')
    parser.add_argument('--web', action='store_true',
                        help='Start web interface instead of CLI mode')
    args = parser.parse_args()
    
    # Initialize NLTK
    initialize_nltk()
    
    # If web flag is set, start the web app
    if args.web:
        start_web_app()
        return
    
    # Load text from file or use sample text
    if args.file:
        # Check if file exists
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            # Create sample file with the same name
            if ensure_sample_file_exists(args.file):
                print(f"Created sample file '{args.file}' for you to use.")
            else:
                print("Using built-in sample text instead.")
                args.file = None
                
    # Read file if provided and exists
    if args.file and os.path.exists(args.file):
        try:
            with open(args.file, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"Successfully loaded text from '{args.file}'")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            print("Using built-in sample text instead.")
            text = get_sample_text()
    else:
        # Use a sample text if no file provided
        text = get_sample_text()
        print("Using built-in sample text.")
    
    # Auto-detect best algorithm if requested
    if args.algorithm == 'auto':
        # Initialize a summarizer temporarily
        temp_summarizer = TextRankSummarizer()
        sentences = temp_summarizer.preprocess_text(text)
        
        # Use TextRank for longer, more complex texts
        # Use TF-IDF for shorter, more factual texts
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        if len(sentences) > 15 or avg_sentence_length > 20:
            algorithm = 'textrank'  # Better for longer, complex texts
        else:
            algorithm = 'tfidf'  # Better for shorter, factual texts
        
        print(f"Auto-detected algorithm: {algorithm}")
    else:
        algorithm = args.algorithm
    
    # Initialize the selected summarizer
    if algorithm == 'textrank':
        summarizer = TextRankSummarizer()
    else:  # tfidf
        summarizer = TFIDFSummarizer()
    
    try:
        # Use adaptive ratio if requested (ratio=0)
        ratio = args.ratio
        if ratio <= 0:
            sentences = summarizer.preprocess_text(text)
            ratio = summarizer.adaptive_ratio(sentences)
            print(f"Using adaptive highlight ratio: {ratio:.2f} ({int(ratio*100)}%)")
        
        # Generate summary
        summary = summarizer.summarize(text, ratio)
        
        # Identify key terms
        key_terms = summarizer.identify_key_terms(text)
        
        # Print results
        print(f"\nOriginal text ({len(text.split())} words):\n")
        print(text)
        
        print(f"\nSummary using {algorithm.upper()} ({len(' '.join([s for s, _ in summary]).split())} words):\n")
        for sentence, score in summary:
            print(f"[Score: {score:.4f}] {sentence}")
        
        print(f"\nKey terms identified: {', '.join(key_terms)}\n")
        
        print(f"\nHighlighted version:\n")
        original_sentences = summarizer.preprocess_text(text)
        for sentence in original_sentences:
            # Check if this sentence is in the summary
            is_highlighted = any(s == sentence for s, _ in summary)
            if is_highlighted:
                print(f"\033[1;33m{sentence}\033[0m")  # Print in yellow and bold
            else:
                print(sentence)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()# AI-Powered Speed Reader Pro (Enhanced Version)