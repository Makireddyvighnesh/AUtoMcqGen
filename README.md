# MCQ Generator

This project is a web application that allows users to upload PDF files, generate multiple-choice questions (MCQs), and send the generated MCQs to a specified email address. The application utilizes Flask, natural language processing (NLP) techniques, and various machine learning models to preprocess the text, generate MCQs, and evaluate them.

## Features

- Upload PDF files and extract text from them.
- Summarize the extracted text using TF-IDF.
- Perform topic modeling using Latent Dirichlet Allocation (LDA).
- Generate MCQs based on the summarized text or user-specified topics.
- Evaluate the generated MCQs using a pre-trained language model.
- Send the generated MCQs as a PDF to the user's email.

## Technologies Used

- Flask
- Flask-WTF
- Flask-Mail
- PyPDF2
- scikit-learn
- nltk
- gensim
- lmql
- pdfkit

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mcq-generator.git
cd mcq-generator
