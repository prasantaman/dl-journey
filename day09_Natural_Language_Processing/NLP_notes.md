## Natural Language Processing (NLP) – Full Detailed Notes

---

### 🔹 What is NLP?

NLP (Natural Language Processing) is a field of AI that enables computers to **understand, interpret, generate, and respond to human languages**.

It combines **linguistics + computer science + machine learning**.

---

### 🔹 Real-life Examples

* Chatbots (e.g., ChatGPT, Alexa)
* Google Translate
* Spam detection
* Voice assistants
* Sentiment analysis
* Autocomplete & spell-check

---

### 🔹 Key Components

| Component                      | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| Text preprocessing             | Cleaning and preparing text                        |
| Tokenization                   | Breaking text into words/tokens                    |
| Stopword removal               | Removing common words (e.g., 'is', 'the')          |
| Stemming                       | Reducing words to root form (e.g., playing → play) |
| Lemmatization                  | Converting to dictionary root form                 |
| POS tagging                    | Identifying parts of speech                        |
| Named Entity Recognition (NER) | Recognizing names, places, dates                   |
| Parsing                        | Analyzing grammatical structure                    |

---

### 🔹 NLP Pipeline

1. **Text Collection** – Collect raw data
2. **Text Cleaning** – Remove noise (punctuation, HTML tags, etc.)
3. **Tokenization** – Split into words
4. **Stopword Removal** – Remove unimportant words
5. **Stemming/Lemmatization** – Normalize words
6. **Feature Extraction** – Convert to numerical form
7. **Modeling** – Apply ML/Deep Learning models
8. **Evaluation** – Accuracy, precision, recall, etc.

---

### 🔹 Feature Extraction Techniques

| Technique       | Description                                    |
| --------------- | ---------------------------------------------- |
| Bag of Words    | Count word frequency                           |
| TF-IDF          | Weigh words based on frequency & rarity        |
| Word2Vec        | Turns words into dense vectors (context-based) |
| GloVe           | Word vectors from global word co-occurrence    |
| BERT Embeddings | Contextual embeddings using transformers       |

---

### 🔹 NLP Tasks

| Task                           | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| Text Classification            | Assign category to text (e.g., spam detection) |
| Sentiment Analysis             | Detect emotion (positive, negative, neutral)   |
| Named Entity Recognition (NER) | Extract proper names, places, etc.             |
| Machine Translation            | Translate text between languages               |
| Text Summarization             | Generate short summary                         |
| Question Answering             | Extract answer from context                    |
| Text Generation                | Generate meaningful text                       |
| POS Tagging                    | Label each word with its part of speech        |

---

### 🔹 Classical ML Models in NLP

* Naive Bayes
* Support Vector Machine (SVM)
* Logistic Regression
* Decision Trees
* Random Forest

---

### 🔹 Deep Learning Models in NLP

| Model       | Usage                           |
| ----------- | ------------------------------- |
| RNN         | Sequence data                   |
| LSTM        | Long-term dependencies          |
| GRU         | Faster LSTM alternative         |
| CNN         | Local feature detection in text |
| Transformer | Attention-based architecture    |
| BERT        | Pre-trained contextual model    |
| GPT         | Text generation, QA             |

---

### 🔹 Text Preprocessing Example (Python)

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is amazing!"
tokens = word_tokenize(text)
filtered = [w for w in tokens if w.lower() not in stopwords.words('english')]
stemmed = [PorterStemmer().stem(w) for w in filtered]
print(stemmed)
```

---

### 🔹 Libraries for NLP

| Library                    | Description                 |
| -------------------------- | --------------------------- |
| NLTK                       | Basic NLP tasks             |
| SpaCy                      | Fast, industrial-grade NLP  |
| Gensim                     | Topic modeling and Word2Vec |
| Scikit-learn               | ML algorithms for NLP       |
| Transformers (HuggingFace) | BERT, GPT, etc.             |

---

### 🔹 Applications in Data Science

* Chatbots and virtual assistants
* Social media sentiment tracking
* Survey analysis
* Text-based recommendation systems
* Legal or medical document analysis

---

### 🔹 Challenges in NLP

* Ambiguity (e.g., "He saw the man with the telescope")
* Sarcasm detection
* Multilingual understanding
* Context awareness
* Slang and informal language

---

### 🔹 Modern Trends in NLP (2024–2025)

* Few-shot and zero-shot learning
* Large Language Models (LLMs) like GPT, BERT, LLaMA
* Prompt engineering
* Retrieval-augmented generation (RAG)
* Multimodal NLP (text + images)
* Responsible and ethical NLP (bias, fairness)

---

### 📌 Summary

* NLP is a bridge between human language and computers
* Key steps: Preprocessing, feature extraction, modeling
* Techniques: ML + DL + Transformers
* Tools: NLTK, SpaCy, BERT, GPT

---

### 📖 Flashcards

```
Q: What is NLP?
A: A field of AI that enables machines to understand and process human language.

Q: What is tokenization?
A: Splitting a sentence into words or tokens.

Q: What is stemming?
A: Reducing words to their base/root form (e.g., 'running' → 'run').

Q: What is TF-IDF?
A: A method to weigh words based on importance in a document and rarity across all documents.

Q: Which model introduced self-attention?
A: Transformer.

Q: What's the difference between stemming and lemmatization?
A: Stemming is rule-based; lemmatization is dictionary-based and more accurate.
```
