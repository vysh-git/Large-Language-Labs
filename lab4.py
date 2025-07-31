# multilingual_qa_system_with_metrics.py
import streamlit as st
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration
)
import pdfplumber
from gtts import gTTS
from io import BytesIO
import logging
import time
import base64
from rouge_score import rouge_scorer
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# IMPROVED CORE FUNCTIONS
# ======================

def extract_text_from_pdf(pdf_file):
    """More robust PDF text extraction"""
    try:
        if pdf_file is None:
            return None
            
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        st.error(f"Failed to extract text: {str(e)}")
        return None

def load_model_with_fallback(model_type):
    """Improved model loading with better error handling"""
    MODEL_MAP = {
        'foundation': {
            'primary': "google/flan-t5-small",
            'fallback': "valhalla/t5-small-qa-qg-hl"
        },
        'indic': {
            'primary': "ai4bharat/indic-bert",
            'fallback': "bert-base-multilingual-cased"
        },
        'intl': {
            'primary': "deepset/xlm-roberta-base-squad2",
            'fallback': "bert-base-multilingual-cased"
        }
    }
    
    for attempt in ['primary', 'fallback']:
        model_name = MODEL_MAP[model_type][attempt]
        try:
            if model_type == 'foundation':
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                return {
                    'tokenizer': tokenizer,
                    'model': model,
                    'type': 'generative',
                    'name': model_name,
                    'status': f"✅ Loaded ({attempt})"
                }
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                return {
                    'tokenizer': tokenizer,
                    'model': model,
                    'type': 'extractive',
                    'name': model_name,
                    'status': f"✅ Loaded ({attempt})"
                }
        except Exception as e:
            logger.warning(f"Failed {attempt} for {model_type}: {e}")
            continue
    
    return {
        'status': f"❌ Failed (all attempts)",
        'name': "No model available"
    }

def generate_answer(context, question, model_info):
    """Generate answer with evaluation metrics"""
    if not context or not question:
        return "Please provide both context and question", 0, 0
    
    try:
        if model_info['type'] == 'generative':
            input_text = f"question: {question} context: {context}"
            input_ids = model_info['tokenizer'](
                input_text, return_tensors="pt"
            ).input_ids.to(model_info['model'].device)
            
            outputs = model_info['model'].generate(input_ids, max_length=200)
            answer = model_info['tokenizer'].decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = model_info['tokenizer'](
                question, context,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model_info['model'].device)
            
            with torch.no_grad():
                outputs = model_info['model'](**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = model_info['tokenizer'].convert_tokens_to_string(
                model_info['tokenizer'].convert_ids_to_tokens(
                    inputs["input_ids"][0][answer_start:answer_end]
                )
            )
        
        # Calculate evaluation metrics
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(context, answer)
        rouge1 = scores['rouge1'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        return answer if answer.strip() else "No answer found in document", rouge1, rougeL
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return f"Error generating answer: {str(e)}", 0, 0

def text_to_speech(text, lang='en'):
    """More reliable TTS with error handling"""
    try:
        if not text or not isinstance(text, str):
            return None
            
        audio = BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(audio)
        audio.seek(0)
        
        # Encode audio for better Streamlit compatibility
        audio_base64 = base64.b64encode(audio.read()).decode('utf-8')
        audio_html = f"""
            <audio autoplay="true" controls style="width:100%">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """
        return audio_html
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ======================
# STREAMLIT UI WITH EVALUATION
# ======================

def main():
    st.set_page_config(page_title="Multilingual QA System", layout="wide")
    st.title("🌍 Multilingual Document QA System")
    
    # Initialize models and evaluation data
    if 'models' not in st.session_state:
        with st.spinner("Loading AI models (this may take a few minutes)..."):
            st.session_state.models = {
                'foundation': load_model_with_fallback('foundation'),
                'indic': load_model_with_fallback('indic'),
                'intl': load_model_with_fallback('intl')
            }
            st.session_state.evaluation = defaultdict(list)
            time.sleep(1)
    
    # Model status dashboard
    with st.expander("MODEL STATUS", expanded=True):
        cols = st.columns(3)
        model_status = {
            'foundation': "FLAN-T5 (English)",
            'indic': "IndicBERT (Hindi)",
            'intl': "CamemBERT (French)"
        }
        
        for i, (model_type, model_info) in enumerate(st.session_state.models.items()):
            with cols[i]:
                st.subheader(model_status[model_type])
                st.write(model_info['status'])
                st.caption(model_info['name'])
    
    # File upload section
    st.sidebar.header("DOCUMENT UPLOAD")
    english_file = st.sidebar.file_uploader("English PDF", type="pdf")
    hindi_file = st.sidebar.file_uploader("Hindi PDF", type="pdf")
    french_file = st.sidebar.file_uploader("French PDF", type="pdf")
    
    # Main QA interface with evaluation tab
    tab1, tab2, tab3, tab4 = st.tabs(["🇬🇧 ENGLISH", "🇮🇳 HINDI", "🇫🇷 FRENCH", "📊 EVALUATION"])
    
    # English QA
    with tab1:
        if english_file and 'model' in st.session_state.models['foundation']:
            context = extract_text_from_pdf(english_file)
            if context:
                question = st.text_area("Ask a question about the English document:", height=100)
                if st.button("Get Answer", key="en_btn"):
                    if question.strip():
                        with st.spinner("Analyzing document..."):
                            answer, rouge1, rougeL = generate_answer(
                                context,
                                question,
                                st.session_state.models['foundation']
                            )
                            st.markdown(f"### Answer:\n{answer}")
                            
                            # Store evaluation results
                            st.session_state.evaluation['english'].append({
                                'question': question,
                                'answer': answer,
                                'rouge1': rouge1,
                                'rougeL': rougeL
                            })
                            
                            audio_html = text_to_speech(answer, 'en')
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.warning("Please enter a question")
    
    # Hindi QA
    with tab2:
        if hindi_file and 'model' in st.session_state.models['indic']:
            context = extract_text_from_pdf(hindi_file)
            if context:
                question = st.text_area("Ask a question about the Hindi document:", height=100)
                if st.button("Get Answer", key="hi_btn"):
                    if question.strip():
                        with st.spinner("दस्तावेज़ का विश्लेषण किया जा रहा है..."):
                            answer, rouge1, rougeL = generate_answer(
                                context,
                                question,
                                st.session_state.models['indic']
                            )
                            st.markdown(f"### उत्तर:\n{answer}")
                            
                            st.session_state.evaluation['hindi'].append({
                                'question': question,
                                'answer': answer,
                                'rouge1': rouge1,
                                'rougeL': rougeL
                            })
                            
                            audio_html = text_to_speech(answer, 'hi')
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.warning("कृपया एक प्रश्न दर्ज करें")
    
    # French QA
    with tab3:
        if french_file and 'model' in st.session_state.models['intl']:
            context = extract_text_from_pdf(french_file)
            if context:
                question = st.text_area("Posez une question sur le document français:", height=100)
                if st.button("Obtenir la réponse", key="fr_btn"):
                    if question.strip():
                        with st.spinner("Analyse du document..."):
                            answer, rouge1, rougeL = generate_answer(
                                context,
                                question,
                                st.session_state.models['intl']
                            )
                            st.markdown(f"### Réponse:\n{answer}")
                            
                            st.session_state.evaluation['french'].append({
                                'question': question,
                                'answer': answer,
                                'rouge1': rouge1,
                                'rougeL': rougeL
                            })
                            
                            audio_html = text_to_speech(answer, 'fr')
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.warning("Veuillez saisir une question")
    
    # Evaluation Metrics Tab
    with tab4:
        st.header("📊 Evaluation Metrics")
        
        # Display metrics for each language
        for lang in ['english', 'hindi', 'french']:
            if st.session_state.evaluation[lang]:
                st.subheader(f"{lang.capitalize()} QA Performance")
                df = pd.DataFrame(st.session_state.evaluation[lang])
                
                # Calculate averages
                avg_rouge1 = df['rouge1'].mean()
                avg_rougeL = df['rougeL'].mean()
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Questions", len(df))
                col2.metric("Average ROUGE-1", f"{avg_rouge1:.3f}")
                col3.metric("Average ROUGE-L", f"{avg_rougeL:.3f}")
                
                # Show detailed results in expandable section
                with st.expander("View Detailed Results"):
                    st.dataframe(df)
                
                # Show score distributions
                st.write("Score Distributions:")
                st.bar_chart(df[['rouge1', 'rougeL']])
            else:
                st.warning(f"No evaluation data available for {lang}")

if __name__ == "__main__":
    main()