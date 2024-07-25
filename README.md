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
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables. Create a `.env` file in the root directory and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Configuration

1. Update the Flask-Mail configuration in `main.py` with your email server settings:

    ```python
    app.config['MAIL_SERVER'] = 'smtp.example.com'
    app.config['MAIL_PORT'] = 465
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USE_SSL'] = True
    app.config['MAIL_USERNAME'] = 'your_email@example.com'
    app.config['MAIL_PASSWORD'] = 'your_email_password'
    ```

2. Make sure you have the required NLP models downloaded. For `nltk`, run the following in a Python shell:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

1. Start the Flask application:

    ```bash
    python main.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Upload a PDF file and specify topics (optional).

4. The application will extract text from the PDF, generate MCQs, and send them to the specified email address as a PDF.

## Project Structure

- `main.py`: The main Flask application file.
- `Preprocessing.py`: Contains classes for PDF cleaning and text summarization.
- `Query.py`: Contains classes for generating and evaluating MCQs.
- `templates/`: Contains the HTML templates for the web interface.
- `static/`: Contains static files like CSS and uploaded files.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [OpenAI](https://www.openai.com/) for providing the language model.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [nltk](https://www.nltk.org/) for natural language processing tools.
- [gensim](https://radimrehurek.com/gensim/) for topic modeling.
- [pdfkit](https://pypi.org/project/pdfkit/) for PDF generation.
