# Career Guidance Chatbot

## 1. Introduction

The goal of this project is to build an intelligent career guidance chatbot that advises users on potential career paths in the technology sector based on their interests. The system leverages natural language processing (NLP) techniques, machine learning classification, and conversational interfaces. The core of the project is a classifier trained on a career guidance Q&A dataset from Hugging Face, while the user interaction is managed via Google Dialogflow. A Flask API serves the classifier, and ngrok is used to expose the local Flask server publicly.

## 2. Dataset Overview

**Dataset:** [Career Guidance Q&A Dataset](https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset)

**Dataset Characteristics:**
- **Content:** Contains questions and answers related to career guidance, including queries about technology sectors, job roles, and career advice.
- **Purpose:** Provides training data for building a classifier that predicts an appropriate technology role based on a user’s input.

## 3. Data Preprocessing and Feature Engineering

Effective preprocessing and feature engineering were critical for extracting meaningful information from the text data. The following techniques were used:

### Preprocessing Steps
- **Stop Words Removal:**  
  Removed common stop words (e.g., "the", "is", "and") to reduce noise and focus on key terms.
- **Stemming and Lemmatization:**  
  - **Stemming:** Reduced words to their base or root form using the Porter Stemmer.
  - **Lemmatization:** Normalized words to their canonical forms using the WordNet lemmatizer.

### Feature Engineering Techniques
- **Bag-of-Words (BoW):**  
  Converted text into a matrix of token counts, representing each document as a vector of word frequencies.
- **n-grams:**  
  Captured contextual information by considering contiguous sequences of words.
- **TF-IDF:**  
  Re-weighted raw frequency counts to emphasize important words (frequent in a document but rare across the corpus).

### Implementation Highlights
- Text was cleaned using regular expressions (removing punctuation and extra whitespace) and normalized to lowercase.
- After tokenization, stemming and lemmatization were applied.
- The feature engineering pipeline experimented with multiple representations (BoW, n-grams, TF-IDF) to select the best-performing method.

## 4. Model Training and Evaluation

Several classifiers were evaluated to identify the best model for predicting a career role:

### Classifiers Explored
- **Naive Bayes:**  
  A probabilistic classifier based on Bayes' theorem.
- **K-Nearest Neighbors (KNN):**  
  A non-parametric classifier that predicts a class based on the majority vote of its nearest neighbors.
- **Random Forest:**  
  An ensemble method that constructs multiple decision trees and outputs the mode of the classes.

### Model Selection Process
- **Evaluation Metric:**  
  F1 Score was used to balance precision and recall.
- **Results:**  
  The Random Forest model combined with the Bag-of-Words representation delivered the best performance in terms of F1 score and overall accuracy.
- **Hyperparameter Tuning:**  
  A grid search with stratified k-fold cross-validation was performed to fine-tune parameters.

## 5. System Integration

### 5.1 Classifier Deployment

The selected model (Random Forest with BoW features) was deployed as a RESTful API using Flask.

- **Flask API Endpoint:**  
  The classifier is served at the `/predict` route, accepting a POST request with user input in JSON format.
- **Input Processing:**  
  The input text is preprocessed (stop words removal, stemming, and lemmatization) and transformed using the BoW vectorizer before prediction.
- **Output:**  
  Returns a JSON response with the predicted role.

### 5.2 Exposing the Classifier with ngrok

Since the Flask API is hosted locally, ngrok is used to create a public URL.

- **ngrok Command:**  
  ```bash
  ngrok http 5000
  ```
- **Usage:** 
The generated public URL (e.g., https://<ngrok-id>.ngrok.io/predict) is used to connect Dialogflow with the classifier API.

  ### 5.3 Dialogflow Chatbot Integration

A Dialogflow agent was created to serve as the user-facing chatbot interface.

- **Intent Design:**  
  An intent (e.g., `CareerAdviceIntent`) was configured with training phrases to capture user input regarding their interests. The parameter `userInterest` is extracted from the user's input.

- **Webhook Fulfillment:**  
  The intent is set to use webhook fulfillment, so Dialogflow forwards the `userInterest` parameter to the classifier API.

- **Response Handling:**  
  The classifier's prediction is returned by the Flask API and formatted into a response (using the `fulfillmentText` field) which Dialogflow then delivers to the user.

---

## 6. Results and Observations

- **Classifier Performance:**  
  The Random Forest model, combined with the BoW features, consistently provided the best predictions. The F1 score and ROC-AUC values during cross-validation indicated a robust model.

- **Chatbot Interaction:**  
  The integration with Dialogflow allowed for a smooth conversational flow. Users are greeted, asked for their interest in technology, and then provided with a tailored career suggestion based on the model's prediction.

- **Deployment Challenges:**  
  - **Endpoint Exposure:** Using ngrok was essential for making the local Flask server accessible, but it requires constant monitoring as free ngrok URLs may change.  
  - **Response Formatting:** Ensuring the Flask API response was compatible with Dialogflow's webhook format was a key step in successful integration.

---

## 7. Challenges and Future Work

### Challenges Faced

- **Data Preprocessing:**  
  Balancing between aggressive cleaning (which might remove valuable information) and retaining sufficient detail to accurately predict the role.

- **Model Selection:**  
  Experimenting with multiple classifiers required careful tuning and evaluation to avoid overfitting and to ensure robust performance across varied inputs.

- **System Integration:**  
  Exposing the Flask API securely via ngrok and ensuring consistent communication with Dialogflow involved several rounds of testing and debugging.

### Future Enhancements

- **Scaling Deployment:**  
  Moving from a local server to a production-ready environment (such as Google Cloud Run or AWS Elastic Beanstalk) to improve reliability and scalability.

- **Additional Models:**  
  Exploring ensemble methods or deep learning models could further enhance prediction accuracy.

- **Enhanced Dialogue Management:**  
  Extending the chatbot’s capabilities to handle follow-up questions and multi-turn conversations would provide a more interactive user experience.

- **User Feedback Loop:**  
  Implementing mechanisms to capture user feedback and continuously update the model could lead to ongoing improvements in the advice given.

  
