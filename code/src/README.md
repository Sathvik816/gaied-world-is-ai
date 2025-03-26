#Project Name: gaied-world-is-ai

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

## 🎯 Introduction
This project extracts content from EML files and classifies emails based on their extracted content. It utilizes a pre-trained Generative AI model for zero-shot and few-shot classification, with plans to fine-tune a loan-specific model.

## Inspiration
The inspiration behind this project seems to be solving real-world email classification challenges, particularly in domains like loan service requests. Key motivations include:
Automating Email Processing
GenAI for Intelligent Classification
Fine-tuning for Domain-Specific Accuracy
Deployment as an API

## What it does
- Extracts email content from EML files
- Applies zero-shot and few-shot classification
- Supports fine-tuning for loan-specific email classification
- Deployable as an API

## How we built it
EML File Processing Module → Extracts content from emails.
GenAI Classification Module → Uses LLMs for email categorization.
API Deployment Module → Serves the classification results via an API.

## Challenges We Faced
Identify the appropriate model for classification
Training the model
Utilize free tools and resources to implement the solution

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Google Colab (recommended for execution)
- Required Python libraries (listed in `requirements.txt`)

## How to checkout
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Upload your EML files to the working directory.

## How to Run
### Running in Google Colab
1. Upload the Colab notebook (`EML_Classification.ipynb`).
2. Upload your EML files to Colab's working directory.
3. Run all cells sequentially.


## Future Enhancements
- Implement fine-tuning with domain-specific datasets
- Improve accuracy with prompt engineering
- Deploy as a REST API

## Tech Stack
Python, Gemini-1.5-pro-latest LLM, FAISS vectordb, sentence-transformer embeddings

## 👥 WorldisAI
- Jayaprakash Baduru
- Pradeep Narayanaswamy
- Thrivikram Chintalapalle
- Sathvik S Gunda
- Bikash Bhanjababu
