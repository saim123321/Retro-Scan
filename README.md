# RetroScan - AI-Powered Text Highlighting Tool

RetroScan is an advanced text analysis application that uses AI algorithms to intelligently highlight the most important sentences in any text. The application helps users save time by focusing on the most relevant content while maintaining full comprehension.

![RetroScan Screenshot](https://placeholder-for-your-screenshot.com/retroscan.png)

## 🚀 Features

- **Dual Algorithm System**: Uses both TextRank and TF-IDF to adapt to different text types
- **Automatic Algorithm Selection**: Intelligently selects the best algorithm based on text structure
- **Adaptive Highlighting**: Dynamically adjusts highlight ratio based on text length
- **Domain Detection**: Recognizes text domain (science, business, technology, etc.) and adapts highlighting
- **Key Term Identification**: Extracts and displays important terms from the text
- **User Highlighting**: Add your own highlights to complement the AI suggestions
- **Interactive UI**: Retro-futuristic neon interface with real-time processing
- **Statistics**: View word counts, highlighting percentages, and other metrics
- **Command Line Interface**: Use as a CLI tool for integration with other workflows

## 📋 Requirements

- Python 3.8+
- Flask
- NumPy
- scikit-learn
- NetworkX
- NLTK

## 🔧 Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/retroscan.git
   cd retroscan
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) If NLTK resources fail to download automatically, run:
   ```
   python -m nltk.downloader punkt stopwords
   ```

## 💻 Usage

### Web Interface

To start the web application:

```
python app.py --web
```

Then open your browser and go to: http://localhost:5000

### Command Line Interface

Process a text file and display highlighted sentences:

```
python app.py --file your_text_file.txt
```

Additional command-line options:

```
python app.py --help
```

### Command-line Arguments

- `--file`: Path to the text file to summarize
- `--algorithm`: Summarization algorithm to use (textrank, tfidf, auto)
- `--ratio`: Proportion of sentences to include (0.0-1.0 or 0 for adaptive)
- `--web`: Start web interface instead of CLI mode

Example:
```
python app.py --file article.txt --algorithm auto --ratio 0.4
```

## 💡 How It Works

RetroScan uses two complementary algorithms to identify important sentences:

1. **TextRank Algorithm**: A graph-based algorithm similar to Google's PageRank that ranks sentences based on their similarity to other sentences in the text.

2. **TF-IDF Algorithm**: A statistical method that evaluates how important a word is to a document in a collection, helping identify sentences with significant terms.

The application also employs several enhancement techniques:
- Positional weighting (first/last sentences often contain important information)
- Key term boosting (sentences containing frequently used terms get higher scores)
- Domain-specific weighting (applies different scoring for different content types)

## 🌟 Web Interface Features

- **Algorithm Selection**: Choose between TextRank, TF-IDF, or Auto-detect
- **Highlight Percentage**: Adjust how much of the text to highlight
- **Custom Highlight Color**: Personalize the highlight color
- **Highlight Toggle**: Switch between showing all text or only highlighted sentences
- **User Highlighting**: Add your own highlights by clicking on sentences
- **Copy Function**: Copy the processed text with highlights
- **Sample Text**: Load a sample text to try the functionality

## 📚 Sample Usage

```python
from app import TextRankSummarizer

# Initialize the summarizer
summarizer = TextRankSummarizer()

# Process some text
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural 
intelligence displayed by animals including humans. AI research has been defined as the field 
of study of intelligent agents, which refers to any system that perceives its environment and 
takes actions that maximize its chance of achieving its goals.
"""

# Get the most important sentences (with adaptive ratio)
summary = summarizer.summarize(text, ratio=0)

# Print the highlighted sentences
for sentence, score in summary:
    print(f"[{score:.2f}] {sentence}")
```

## 🔄 Deployment

The application can be deployed using:

1. **Render**: Create a web service with the following settings:
   - Build command: `pip install -r requirements.txt && python -m nltk.downloader punkt stopwords`
   - Start command: `gunicorn wsgi:app --timeout 120`

2. **Heroku**: Use the included Procfile 

3. **Docker**: A Dockerfile is provided for containerized deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- NLTK team for their natural language processing tools
- NetworkX project for graph algorithms
- scikit-learn for machine learning capabilities
- Flask for the web framework
