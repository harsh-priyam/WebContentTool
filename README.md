# README for Web Scraper Chatbot Application

## Overview
This application scrapes a webpage, extracts text, creates embeddings using OpenAI’s API, and allows you to query the content using a chatbot.

---

## Prerequisites
1. **Python Version**: Ensure Python 3.8 or higher is installed.
2. **OpenAI API Key**: You must have an OpenAI API key saved in a `.env` file.

---

## Setup Instructions

1. **Prepare Environment**:
   - Create a new Conda environment:
     ```bash
     conda create -n scraper-chatbot-env python=3.11 -y
     conda activate scraper-chatbot-env
     ```

2. **Install Dependencies**:
   - Use the provided `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

3. **Add OpenAI API Key**:
   - Create a `.env` file in the same directory as the script.
   - Add your API key in the following format:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Run the Script**:
   - Execute the Python file:
     ```bash
     streamlit run app.py
     ```

---

## Usage

1. **Enter URL**:
   - When prompted, provide the URL of the webpage to scrape.

2. **Ask Queries**:
   - Input your query related to the webpage content.
   - The chatbot will respond with professional answers.

---

## Notes
- Ensure the URL is accessible and doesn’t block scraping.
- Handle the API key securely and avoid sharing it publicly.

---

## Troubleshooting
1. **API Key Errors**:
   - Confirm the `.env` file is in the correct directory and contains the key.

2. **Dependency Issues**:
   - Reinstall dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

