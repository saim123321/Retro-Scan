from fp import start_web_app, initialize_nltk

# Initialize NLTK resources on startup
initialize_nltk()

# Create the Flask app (without running it)
app = None

if __name__ == "__main__":
    start_web_app()
else:
    # Import the Flask app for Gunicorn
    import fp
    from flask import Flask
    
    # Create a modified version of the start_web_app function that returns the app
    def create_app():
        app = Flask(__name__)
        
        # Initialize NLTK
        initialize_nltk()
        
        # Initialize summarizers
        from fp import TextRankSummarizer, TFIDFSummarizer
        text_rank = TextRankSummarizer()
        tfidf = TFIDFSummarizer()
        
        # Add routes
        @app.route('/')
        def index():
            from flask import render_template_string
            return render_template_string(fp.HTML_TEMPLATE)
        
        @app.route('/summarize', methods=['POST'])
        def summarize():
            from flask import request, jsonify
            import traceback
            
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
        
        return app
    
    # Create the app instance
    app = create_app()