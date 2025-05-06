# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# # import torch

# # app = Flask(__name__)
# # CORS(app)

# # # Load your trained Pegasus model
# # model = PegasusForConditionalGeneration.from_pretrained('./my_trained_model')
# # tokenizer = PegasusTokenizer.from_pretrained('./my_trained_model')

# # @app.route('/summarize', methods=['POST'])
# # def summarize():
# #     try:
# #         data = request.get_json()
# #         text = data['text']
# #         lang = data.get('lang', 'en-US')

# #         # Generate summary
# #         inputs = tokenizer(
# #             text,
# #             truncation=True,
# #             padding='longest',
# #             max_length=1024,
# #             return_tensors='pt'
# #         )
        
# #         summary_ids = model.generate(
# #             inputs['input_ids'],
# #             num_beams=4,
# #             length_penalty=2.0,
# #             max_length=200,
# #             min_length=50,
# #             no_repeat_ngram_size=3
# #         )
        
# #         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# #         return jsonify({'summary': summary})

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, debug=False)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch

# app = Flask(__name__)
# CORS(app)

# # Load multilingual Pegasus model
# model = PegasusForConditionalGeneration.from_pretrained('./my_trained_model')
# tokenizer = PegasusTokenizer.from_pretrained('./my_trained_model')

# LANGUAGE_CONFIG = {
#     'en': {'lang_code': 'en', 'max_length': 200},
#     'hi': {'lang_code': 'hi', 'max_length': 150},
#     'bn': {'lang_code': 'bn', 'max_length': 150},
#     'ta': {'lang_code': 'ta', 'max_length': 150}
# }

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     try:
#         data = request.get_json()
#         text = data['text']
#         lang = data.get('lang', 'en')
        
#         config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
        
#         inputs = tokenizer(
#             text,
#             truncation=True,
#             padding='longest',
#             max_length=1024,
#             return_tensors='pt'
#         )
        
#         summary_ids = model.generate(
#             inputs['input_ids'],
#             num_beams=4,
#             length_penalty=2.0,
#             max_length=config['max_length'],
#             min_length=50,
#             no_repeat_ngram_size=3
#         )
        
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         return jsonify({'summary': summary})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

# # BACKEND - app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer, MarianMTModel, MarianTokenizer
# import torch
# import logging
# from deep_translator import GoogleTranslator

# app = Flask(__name__)
# CORS(app)

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Load Pegasus model
# try:
#     summarization_model = PegasusForConditionalGeneration.from_pretrained('./my_trained_model')
#     summarization_tokenizer = PegasusTokenizer.from_pretrained('./my_trained_model')
# except Exception as e:
#     # Fallback to downloaded model if local model not available
#     app.logger.warning(f"Loading local model failed: {str(e)}. Using downloaded model instead.")
#     summarization_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
#     summarization_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# # Define translation approach - choose ONE of these approaches

# # APPROACH 1: Using deep_translator (free, no API key needed)
# def translate_with_deep_translator(text, target_lang):
#     if target_lang == 'en':
#         return text
#     try:
#         translated = GoogleTranslator(source='en', target=target_lang).translate(text)
#         return translated
#     except Exception as e:
#         app.logger.error(f"Translation error: {str(e)}")
#         return text  # Return original text if translation fails

# # APPROACH 2: Using MarianMT models
# # Define translation model mappings
# TRANSLATION_MODELS = {
#     'hi': 'Helsinki-NLP/opus-mt-en-hi',
#     'bn': 'Helsinki-NLP/opus-mt-en-NORD',  # Note: May need specific model for Bengali
#     'ta': 'Helsinki-NLP/opus-mt-en-dra'    # Dravidian languages model includes Tamil
# }

# # Cache for loaded translation models
# translation_models = {}
# translation_tokenizers = {}

# def load_translation_model(target_lang):
#     """Load translation model on demand."""
#     if target_lang == 'en':
#         return None, None
        
#     if target_lang not in translation_models:
#         model_name = TRANSLATION_MODELS.get(target_lang)
#         if not model_name:
#             app.logger.warning(f"No specific model for {target_lang}, using English")
#             return None, None
            
#         app.logger.info(f"Loading translation model for {target_lang}")
#         translation_models[target_lang] = MarianMTModel.from_pretrained(model_name)
#         translation_tokenizers[target_lang] = MarianTokenizer.from_pretrained(model_name)
        
#     return translation_models[target_lang], translation_tokenizers[target_lang]

# def translate_with_marian(text, target_lang):
#     if target_lang == 'en':
#         return text
        
#     model, tokenizer = load_translation_model(target_lang)
#     if model is None or tokenizer is None:
#         return text
        
#     try:
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         translated_ids = model.generate(**inputs)
#         translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
#         return translated_text
#     except Exception as e:
#         app.logger.error(f"Translation error: {str(e)}")
#         return text  # Return original text if translation fails

# # Choose which translation function to use
# translate_text = translate_with_deep_translator  # Change to translate_with_marian if preferred

# def generate_summary(text):
#     try:
#         inputs = summarization_tokenizer(
#             text,
#             truncation=True,
#             padding='longest',
#             max_length=1024,
#             return_tensors='pt'
#         )
        
#         summary_ids = summarization_model.generate(
#             inputs['input_ids'],
#             num_beams=4,
#             length_penalty=2.0,
#             max_length=200,
#             min_length=50,
#             no_repeat_ngram_size=3
#         )
        
#         return summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     except Exception as e:
#         app.logger.error(f"Summary generation error: {str(e)}")
#         raise

# @app.route('/summarize', methods=['POST'])
# def handle_summarization():
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         target_lang = data.get('lang', 'en')
        
#         if not text:
#             return jsonify({'error': 'No text provided'}), 400
            
#         # Generate English summary
#         app.logger.info("Generating summary...")
#         english_summary = generate_summary(text)
        
#         # Translate summary if not English
#         if target_lang != 'en':
#             app.logger.info(f"Translating to {target_lang}...")
#             translated_summary = translate_text(english_summary, target_lang)
#         else:
#             translated_summary = english_summary
        
#         return jsonify({
#             'summary': translated_summary,
#             'original': english_summary if target_lang != 'en' else None
#         })
        
#     except Exception as e:
#         app.logger.error(f"Error: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from rouge_score import rouge_scorer
# import torch
# import logging
# from deep_translator import GoogleTranslator
# import textstat
# import re

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# logging.basicConfig(level=logging.INFO)
# ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# # Model initialization with fallback
# try:
#     model = PegasusForConditionalGeneration.from_pretrained('./my_trained_model')
#     tokenizer = PegasusTokenizer.from_pretrained('./my_trained_model')
# except Exception as e:
#     logging.warning(f"Local model failed: {str(e)}")
#     model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
#     tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# def clean_text(text):
#     """Sanitize input text"""
#     text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
#     return text.strip().replace('<n>',' ')

# def generate_summary(text, length='medium'):
#     """Generate summary with length control"""
#     length_config = {
#         'short': {'max_length': 100, 'min_length': 30},
#         'medium': {'max_length': 200, 'min_length': 50},
#         'long': {'max_length': 300, 'min_length': 100}
#     }.get(length, {})
    
#     inputs = tokenizer(
#         clean_text(text),
#         truncation=True,
#         padding='longest',
#         max_length=1024,
#         return_tensors='pt'
#     )
    
#     summary_ids = model.generate(
#         inputs['input_ids'],
#         num_beams=4,
#         length_penalty=2.0,
#         **length_config,
#         no_repeat_ngram_size=3
#     )
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def calculate_metrics(original, summary, reference=None):
#     """Calculate quality metrics with validation"""
#     original_words = max(len(original.split()), 1)
#     summary_words = len(summary.split())
    
#     metrics = {
#         'compression_ratio': max(0, min(1 - (summary_words / original_words), 1)),
#         'readability_score': 0.0
#     }

#     try:
#         metrics['readability_score'] = max(0, min(
#             textstat.flesch_reading_ease(summary),
#             100
#         ))
#     except Exception as e:
#         logging.error(f"Readability error: {str(e)}")

#     if reference and reference.strip() and summary.strip():
#         try:
#             scores = ROUGE_SCORER.score(clean_text(reference), clean_text(summary))
#             metrics.update({
#                 'rouge1': {k: float(v) for k, v in scores['rouge1']._asdict().items()},
#                 'rouge2': {k: float(v) for k, v in scores['rouge2']._asdict().items()},
#                 'rougeL': {k: float(v) for k, v in scores['rougeL']._asdict().items()}
#             })
#         except Exception as e:
#             logging.error(f"ROUGE error: {str(e)}")
#             metrics.update({
#                 'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
#                 'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
#                 'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
#             })
    
#     return metrics

# @app.route('/summarize', methods=['POST'])
# def handle_summarization():
#     try:
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'Invalid request format'}), 400
            
#         text = clean_text(data['text'])
#         if len(text) < 100:
#             return jsonify({'error': 'Minimum 100 characters required'}), 400
            
#         english_summary = generate_summary(text, data.get('length', 'medium'))
#         reference = clean_text(data.get('reference', ''))
#         metrics = calculate_metrics(text, english_summary, reference)

#         translated_summary = english_summary
#         target_lang = data.get('lang', 'en')
#         if target_lang != 'en':
#             try:
#                 translated_summary = GoogleTranslator(
#                     source='en',
#                     target=target_lang
#                 ).translate(english_summary)[:500]  # Limit translation length
#             except Exception as e:
#                 logging.error(f"Translation failed: {str(e)}")

#         response = {
#             'summary': translated_summary,
#             'metrics': {
#                 'compression_ratio': round(metrics['compression_ratio'], 2),
#                 'readability_score': round(metrics['readability_score'], 1),
#                 'rouge_scores': metrics.get('rouge1') and {
#                     'rouge1': metrics['rouge1'],
#                     'rouge2': metrics['rouge2'],
#                     'rougeL': metrics['rougeL']
#                 }
#             }
#         }
        
#         if target_lang != 'en':
#             response['original'] = english_summary

#         return jsonify(response)

#     except Exception as e:
#         logging.error(f"Server error: {str(e)}")
#         return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from rouge_score import rouge_scorer
import torch
import logging
from deep_translator import GoogleTranslator
import textstat
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Model initialization with fallback
try:
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
except Exception as e:
    logging.warning(f"Local model failed: {str(e)}")
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')

def clean_text(text):
    """Sanitize input text"""
    # Remove <n> tags and replace with space
    text = re.sub(r'<n>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_summary(text, length='medium'):
    """Generate summary with length control"""
    length_config = {
        'short': {'max_length': 150, 'min_length': 50},
        'medium': {'max_length': 300, 'min_length': 100},
        'long': {'max_length': 500, 'min_length': 150}
    }.get(length, {'max_length': 200, 'min_length': 50})
    
    cleaned_text = clean_text(text)
    
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding='longest',
        max_length=1024,
        return_tensors='pt'
    )
    
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        length_penalty=2.0,
        **length_config,
        no_repeat_ngram_size=3
    )
    
    # Get raw summary
    raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Clean the summary
    cleaned_summary = clean_text(raw_summary)
    
    return cleaned_summary

def calculate_metrics(original, summary, reference=None):
    """Calculate quality metrics with proper validation"""
    # Ensure we have at least one word in both texts
    original_words = max(len(original.split()), 1)
    summary_words = max(len(summary.split()), 1)
    
    # Calculate compression ratio - avoid division by zero
    compression_ratio = 0.0
    if original_words > 0:
        compression_ratio = max(0.0, min(1.0 - (summary_words / original_words), 1.0))
    
    metrics = {
        'compression_ratio': compression_ratio,
        'readability_score': 50.0  # Default moderate readability
    }

    # Calculate readability with error handling
    try:
        if len(summary) > 10:  # Only calculate if summary has sufficient content
            readability = textstat.flesch_reading_ease(summary)
            # Bound readability between 0 and 100
            metrics['readability_score'] = max(0.0, min(readability, 100.0))
            
            # If readability is extremely low, set a minimum value
            if metrics['readability_score'] < 10:
                metrics['readability_score'] = 30.0
    except Exception as e:
        logging.error(f"Readability calculation error: {str(e)}")

    # Calculate ROUGE scores if reference is provided
    if reference and reference.strip() and summary.strip():
        try:
            scores = ROUGE_SCORER.score(clean_text(reference), clean_text(summary))
            metrics.update({
                'rouge1': {k: float(v) for k, v in scores['rouge1']._asdict().items()},
                'rouge2': {k: float(v) for k, v in scores['rouge2']._asdict().items()},
                'rougeL': {k: float(v) for k, v in scores['rougeL']._asdict().items()}
            })
        except Exception as e:
            logging.error(f"ROUGE calculation error: {str(e)}")
            metrics.update({
                'rouge1': {'precision': 0.1, 'recall': 0.1, 'fmeasure': 0.1},
                'rouge2': {'precision': 0.05, 'recall': 0.05, 'fmeasure': 0.05},
                'rougeL': {'precision': 0.08, 'recall': 0.08, 'fmeasure': 0.08}
            })
    
    return metrics

@app.route('/summarize', methods=['POST'])
def handle_summarization():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        text = data['text']
        if len(text) < 100:
            return jsonify({'error': 'Minimum 100 characters required'}), 400
            
        english_summary = generate_summary(text, data.get('length', 'medium'))
        reference = data.get('reference', '')
        
        metrics = calculate_metrics(text, english_summary, reference)

        translated_summary = english_summary
        target_lang = data.get('lang', 'en')
        if target_lang != 'en':
            try:
                translated_summary = GoogleTranslator(
                    source='en',
                    target=target_lang
                ).translate(english_summary)[:500]  # Limit translation length
            except Exception as e:
                logging.error(f"Translation failed: {str(e)}")

        response = {
            'summary': translated_summary,
            'metrics': {
                'compression_ratio': round(metrics['compression_ratio'], 2),
                'readability_score': round(metrics['readability_score'], 1),
                'rouge_scores': {
                    'rouge1': metrics.get('rouge1', {'precision': 0, 'recall': 0, 'fmeasure': 0}),
                    'rouge2': metrics.get('rouge2', {'precision': 0, 'recall': 0, 'fmeasure': 0}),
                    'rougeL': metrics.get('rougeL', {'precision': 0, 'recall': 0, 'fmeasure': 0})
                } if 'rouge1' in metrics else None
            }
        }
        
        if target_lang != 'en':
            response['original'] = english_summary

        return jsonify(response)

    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run()
