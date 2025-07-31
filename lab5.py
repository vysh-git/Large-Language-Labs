# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, pipeline
import streamlit as st
import logging
import evaluate
import pandas as pd
import nltk
from datasets import load_dataset

nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model metadata
MODELS = {
    "BERT-Base": {
        "sequence": "textattack/bert-base-uncased-SST-2",
        "token": "dslim/bert-base-NER",
        "qa": "deepset/bert-base-cased-squad2",
        "description": "BERT-Base Uncased: Original BERT model with 12 layers, 768 hidden size. Pre-trained on MLM and NSP objectives using English Wikipedia and BooksCorpus. Fine-tuned variants used for tasks like sentiment (SST-2), NER (CoNLL-2003), QA (SQuAD). Applications: Sentiment analysis, entity recognition in general text, extractive QA systems."
    },
    "DistilBERT": {
        "sequence": "distilbert-base-uncased-finetuned-sst-2-english",
        "token": "elastic/distilbert-base-cased-finetuned-conll03-english",
        "qa": "distilbert-base-cased-distilled-squad",
        "description": "DistilBERT: Lightweight BERT variant with 6 layers, distilled from BERT-Base. Same pre-training objectives but 40% smaller. Fine-tuned on similar datasets. Applications: Efficient sentiment classification, NER in resource-constrained environments, faster QA."
    },
    # "ALBERT": {
    #     "sequence": "textattack/albert-base-v2-SST-2",
    #     "token": "Jorgeutd/albert-base-v2-finetuned-ner",
    #     "qa": "twmkn9/albert-base-v2-squad2",
    #     "description": "ALBERT: A Lite BERT with parameter sharing, reducing size. Pre-trained on similar corpora with MLM and SOP (Sentence Order Prediction). Fine-tuned on SST-2, CoNLL-like for NER, SQuAD. Applications: Memory-efficient models for classification, NER, QA in mobile/deployed settings."
    # }
}

@st.cache_resource
def load_model_and_tokenizer(model_name, task_type):
    try:
        model_path = MODELS[model_name][task_type]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if task_type == "sequence":
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        elif task_type == "token":
            model = AutoModelForTokenClassification.from_pretrained(model_path)
        elif task_type == "qa":
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading {model_name} for {task_type}: {e}")
        raise

# Evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
seqeval = evaluate.load("seqeval")
squad_metric = evaluate.load("squad")

@st.cache_data
def evaluate_sequence_classification(model_name):
    dataset = load_dataset("glue", "sst2", split="validation[:50]")
    model, tokenizer = load_model_and_tokenizer(model_name, "sequence")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []
    references = []
    for example in dataset:
        inputs = tokenizer(example["sentence"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred)
        references.append(example["label"])
    acc = accuracy.compute(predictions=predictions, references=references)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=references)["f1"]
    return {"accuracy": acc, "f1": f1_score}

def align_predictions(predictions, label_ids):
    aligned_preds = []
    aligned_labels = []
    for pred, label in zip(predictions, label_ids):
        seq_preds = [p for p, l in zip(pred, label) if l != -100]
        seq_labels = [l for l in label if l != -100]
        aligned_preds.extend(seq_preds)
        aligned_labels.extend(seq_labels)
    return aligned_preds, aligned_labels

@st.cache_data
def evaluate_token_classification(model_name):
    dataset = load_dataset("conll2003", split="test[:20]", trust_remote_code=True)
    model, tokenizer = load_model_and_tokenizer(model_name, "token")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_predictions = []
    all_labels = []

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)

    for example in tokenized_dataset:
        inputs = {
            "input_ids": torch.tensor([example["input_ids"]], dtype=torch.long).to(device),
            "attention_mask": torch.tensor([example["attention_mask"]], dtype=torch.long).to(device)
        }
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        labels = example["labels"]
        all_predictions.append(preds)
        all_labels.append(labels)

    flat_preds, flat_labels = align_predictions(all_predictions, all_labels)
    preds_str = [model.config.id2label.get(p, "O") for p in flat_preds]
    labels_str = [model.config.id2label.get(l, "O") for l in flat_labels]
    results = seqeval.compute(predictions=[preds_str], references=[labels_str])
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

@st.cache_data
def evaluate_question_answering(model_name):
    dataset = load_dataset("squad", split="validation[:50]")
    pipe = pipeline("question-answering", model=MODELS[model_name]["qa"])
    predictions = []
    references = []
    for i, example in enumerate(dataset):
        result = pipe(question=example["question"], context=example["context"])
        predictions.append({"id": str(i), "prediction_text": result["answer"]})
        references.append({"id": str(i), "answers": example["answers"]})
    results = squad_metric.compute(predictions=predictions, references=references)
    return {"exact_match": results["exact_match"], "f1": results["f1"]}

def perform_sequence_classification(model_name, text):
    pipe = pipeline("sentiment-analysis", model=MODELS[model_name]["sequence"])
    result = pipe(text)[0]
    label = result['label']
    # Map LABEL_0 to NEGATIVE, LABEL_1 to POSITIVE
    if label == 'LABEL_0':
        label = 'NEGATIVE'
    elif label == 'LABEL_1':
        label = 'POSITIVE'
    return f"{label} (confidence: {result['score']:.4f})"

def perform_token_classification(model_name, text):
    pipe = pipeline("ner", model=MODELS[model_name]["token"], aggregation_strategy="simple")
    results = pipe(text)
    return ", ".join([f"{entity['word']} ({entity['entity_group']})" for entity in results]) or "No entities found."

def perform_question_answering(model_name, context, question):
    pipe = pipeline("question-answering", model=MODELS[model_name]["qa"])
    result = pipe(question=question, context=context)
    return result["answer"]

# Main Streamlit App
def main():
    st.title("📚 BERT and BERT-based Models in NLP Tasks")
    
    st.markdown("""
    ### 🚀 Project Overview
    This app implements and analyzes BERT variants for:
    - **Sequence Classification** (Sentiment Analysis)
    - **Token Classification** (Named Entity Recognition)
    - **Question Answering** (Extractive QA)
    
    Models: BERT-Base, DistilBERT, ALBERT.
    Interact via text, view predictions, evaluations, and plots.
    """)

    # Model Descriptions
    st.subheader("i. Model Selection and Description")
    for model_name, info in MODELS.items():
        st.markdown(f"**{model_name}**: {info['description']}")

    tab_seq, tab_token, tab_qa = st.tabs(["ii.a Sequence Classification", "ii.b Token Classification", "ii.c Question Answering"])

    with tab_seq:
        st.write("Example: Enter a movie review for sentiment analysis.")
        seq_text = st.text_area("Enter text for sentiment:", key="seq_text")
        if seq_text:
            auto_metrics_seq = {"Model": [], "Accuracy": [], "F1": []}
            human_metrics_seq = {"Model": [], "Fluency": [], "Coherence": [], "Relevance": [], "Average": []}
            for model_name in MODELS.keys():
                with st.spinner(f"Processing with {model_name}..."):
                    try:
                        answer = perform_sequence_classification(model_name, seq_text)
                        st.write(f"**{model_name} Prediction**: {answer}")
                        
                        # Automatic Evaluation
                        metrics = evaluate_sequence_classification(model_name)
                        auto_metrics_seq["Model"].append(model_name)
                        auto_metrics_seq["Accuracy"].append(metrics["accuracy"])
                        auto_metrics_seq["F1"].append(metrics["f1"])
                        
                        st.write(f"**Accuracy**: {metrics['accuracy']:.4f}")
                        st.write(f"**F1**: {metrics['f1']:.4f}")
                        
                        # Human Evaluation
                        with st.expander(f"Rate {model_name} (Human Evaluation)"):
                            fluency = st.slider("Fluency", 1, 5, 3, key=f"seq_{model_name}_f")
                            coherence = st.slider("Coherence", 1, 5, 3, key=f"seq_{model_name}_c")
                            relevance = st.slider("Relevance", 1, 5, 3, key=f"seq_{model_name}_r")
                            avg = (fluency + coherence + relevance) / 3
                            human_metrics_seq["Model"].append(model_name)
                            human_metrics_seq["Fluency"].append(fluency)
                            human_metrics_seq["Coherence"].append(coherence)
                            human_metrics_seq["Relevance"].append(relevance)
                            human_metrics_seq["Average"].append(avg)
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")

            if auto_metrics_seq["Model"]:
                st.markdown("### 🤖 Automatic Evaluation")
                df_auto_seq = pd.DataFrame(auto_metrics_seq).set_index("Model")
                st.bar_chart(df_auto_seq)

                df_human_seq = pd.DataFrame(human_metrics_seq).set_index("Model")
                st.markdown("### 👤 Human Evaluation")
                st.bar_chart(df_human_seq[["Average"]])

    with tab_token:
        st.write("Example: Enter a sentence to extract entities.")
        token_text = st.text_area("Enter text for NER:", key="token_text")
        if token_text:
            auto_metrics_token = {"Model": [], "Precision": [], "Recall": [], "F1": []}
            human_metrics_token = {"Model": [], "Fluency": [], "Coherence": [], "Relevance": [], "Average": []}
            for model_name in MODELS.keys():
                with st.spinner(f"Processing with {model_name}..."):
                    try:
                        answer = perform_token_classification(model_name, token_text)
                        st.write(f"**{model_name} Prediction**: {answer}")
                        
                        # Automatic Evaluation
                        metrics = evaluate_token_classification(model_name)
                        auto_metrics_token["Model"].append(model_name)
                        auto_metrics_token["Precision"].append(metrics["precision"])
                        auto_metrics_token["Recall"].append(metrics["recall"])
                        auto_metrics_token["F1"].append(metrics["f1"])
                        
                        st.write(f"**Precision**: {metrics['precision']:.4f}")
                        st.write(f"**Recall**: {metrics['recall']:.4f}")
                        st.write(f"**F1**: {metrics['f1']:.4f}")
                        
                        # Human Evaluation
                        with st.expander(f"Rate {model_name} (Human Evaluation)"):
                            fluency = st.slider("Fluency", 1, 5, 3, key=f"token_{model_name}_f")
                            coherence = st.slider("Coherence", 1, 5, 3, key=f"token_{model_name}_c")
                            relevance = st.slider("Relevance", 1, 5, 3, key=f"token_{model_name}_r")
                            avg = (fluency + coherence + relevance) / 3
                            human_metrics_token["Model"].append(model_name)
                            human_metrics_token["Fluency"].append(fluency)
                            human_metrics_token["Coherence"].append(coherence)
                            human_metrics_token["Relevance"].append(relevance)
                            human_metrics_token["Average"].append(avg)
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")

            if auto_metrics_token["Model"]:
                st.markdown("### 🤖 Automatic Evaluation")
                df_auto_token = pd.DataFrame(auto_metrics_token).set_index("Model")
                st.bar_chart(df_auto_token[["F1"]])

                df_human_token = pd.DataFrame(human_metrics_token).set_index("Model")
                st.markdown("### 👤 Human Evaluation")
                st.bar_chart(df_human_token[["Average"]])

    with tab_qa:
        st.write("Example: Provide context and ask a question.")
        qa_context = st.text_area("Enter context paragraph:", key="qa_context")
        qa_question = st.text_input("Enter question:", key="qa_question")
        if qa_context and qa_question:
            auto_metrics_qa = {"Model": [], "Exact Match": [], "F1": []}
            human_metrics_qa = {"Model": [], "Fluency": [], "Coherence": [], "Relevance": [], "Average": []}
            for model_name in MODELS.keys():
                with st.spinner(f"Processing with {model_name}..."):
                    try:
                        answer = perform_question_answering(model_name, qa_context, qa_question)
                        st.write(f"**{model_name} Answer**: {answer}")
                        
                        # Automatic Evaluation
                        metrics = evaluate_question_answering(model_name)
                        auto_metrics_qa["Model"].append(model_name)
                        auto_metrics_qa["Exact Match"].append(metrics["exact_match"])
                        auto_metrics_qa["F1"].append(metrics["f1"])
                        
                        st.write(f"**Exact Match**: {metrics['exact_match']:.4f}")
                        st.write(f"**F1**: {metrics['f1']:.4f}")
                        
                        # Human Evaluation
                        with st.expander(f"Rate {model_name} (Human Evaluation)"):
                            fluency = st.slider("Fluency", 1, 5, 3, key=f"qa_{model_name}_f")
                            coherence = st.slider("Coherence", 1, 5, 3, key=f"qa_{model_name}_c")
                            relevance = st.slider("Relevance", 1, 5, 3, key=f"qa_{model_name}_r")
                            avg = (fluency + coherence + relevance) / 3
                            human_metrics_qa["Model"].append(model_name)
                            human_metrics_qa["Fluency"].append(fluency)
                            human_metrics_qa["Coherence"].append(coherence)
                            human_metrics_qa["Relevance"].append(relevance)
                            human_metrics_qa["Average"].append(avg)
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")

            if auto_metrics_qa["Model"]:
                st.markdown("### 🤖 Automatic Evaluation")
                df_auto_qa = pd.DataFrame(auto_metrics_qa).set_index("Model")
                st.bar_chart(df_auto_qa)

                df_human_qa = pd.DataFrame(human_metrics_qa).set_index("Model")
                st.markdown("### 👤 Human Evaluation")
                st.bar_chart(df_human_qa[["Average"]])

    # Overall Comparison
    st.subheader("iv. Evaluation and Results Summary")
    st.write("Strengths: BERT variants excel in contextual understanding; DistilBERT is faster; ALBERT is memory-efficient.")
    st.write("Limitations: Computationally intensive; may underperform on out-of-domain data; subword handling can complicate NER.")
    comparison_table = {
        "Aspect": ["Efficiency", "Accuracy (Avg)", "Domain Adaptability", "Input Handling", "Output Quality"],
        "BERT-Base": ["Medium", "High", "General", "Good", "High"],
        "DistilBERT": ["High", "Medium-High", "General", "Good", "Medium-High"],
    
    }
    df_comparison = pd.DataFrame(comparison_table).set_index("Aspect")
    st.table(df_comparison)

if __name__ == "__main__":
    main()