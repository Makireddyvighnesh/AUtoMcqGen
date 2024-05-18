
from flask import Flask, render_template, jsonify, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from flask_mail import Mail, Message

from sklearn.feature_extraction.text import TfidfVectorizer
import time
import lmql
import sys
sys.path.append(r'C:\Users\vighnesh\Documents\DataScience')
from Preprocessing_text_MCQ import TextSummarizer, TopicModeling, PDFCleaner
from LMQL_Q_A import GenerateMCQ,EvaluateMCQ
import pdfkit
openai_api_key = os.getenv("OPENAI_API_KEY")
# Assuming the notebook is named "my_notebook.ipynb"
# %run "C:/path/to/your/notebook/my_notebook.ipynb"

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MAIL_SERVER'] = 'smtp.example.com'  # Change to your email server
app.config['MAIL_PORT'] = 465  # Change to your email server's port
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'makireddyvighnesh@example.com'  # Change to your email address
app.config['MAIL_PASSWORD'] = 'nannaamma@7389'  # Change to your email password
mail = Mail(app)

def initialize_model():
    model = lmql.model("local:llama.cpp:zephyr-7b-beta.Q4_K_M.gguf", tokenizer='HuggingFaceH4/zephyr-7b-beta', n_ctx =4096)
    return model


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        file.save(file_path)
        pdf_cleaner = PDFCleaner(filename=file_path)
        print("inside")
        num_q_each = 1
        # Spliting user specifies topics with deliminator as comma
        if request.form['topics']:
            user_topics = request.form['topics'].split(',')
            num_q_each = user_topics[-1]
            user_topics = user_topics[0:-1]
        mail = request.form['mail']
        print(f"User mail: {mail}")
        
        # Extracting text feom document
        all_text = pdf_cleaner.extract_text(5,90)
        sentences = pdf_cleaner.chunk_text(chunk_size=5000)

        # Calculating sentence score using TF-IDF matrix
        text_summarizer = TextSummarizer()
        preprocessed_corpus = text_summarizer.preprocess_text(sentences)
        tfidf_vectorizer = TfidfVectorizer()
        print(len(preprocessed_corpus))
        print(preprocessed_corpus)
        tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_corpus)
        print(tfidf_matrix)
        num_sentences = int(len(sentences)*0.5)  # Select 25% of sentences as summary
        summary = text_summarizer.generate_summary(document=preprocessed_corpus, tfidf_matrix=tfidf_matrix, num_sentences=num_sentences)
        print("Summary:")
        print(len(summary))
        print(summary)

        flag = False
        if request.form['topics']:
            # Grouping large text into topics using Latent Dirichlet Allocation(LDA) a topic-modeling algorithm
            lda_model = TopicModeling(sentences)
            lda_model.preprocess_text()
            num_topics = lda_model.visualize_coherence(start=1, stop=20, step=1)
            lda_model.build_lda_model(num_topics)
            topics = lda_model.show_topics(num_topics, num_words=100)
            # user_topics = [["variance"], ["bias"], ["machine"], ["learning"], ["metrics"], ["precision"]]
            print(user_topics)
            print("Topics given by Topic modeling are ", topics)
            user_topics, relevant_topics = lda_model.find_most_relevant_topics_for_user_topics(user_topics)
            relevant_text = [topics[index] for index in relevant_topics]
            print(user_topics, relevant_topics)
            for i in range(len(relevant_topics)):
                relevant_text[i]= ' '.join(relevant_text[i])
            topic_mapping = dict(zip(user_topics, relevant_text))
            print(topic_mapping)
            flag=True

        
        # Generating MCQs using LMQL which used to interact with LLMs
        
        lmql_generator = GenerateMCQ(n=num_q_each)
        size = 4000
        text = topic_mapping if flag else summary
        print(text)
        print("HElooooooooooooooo")
        if flag:
            for value in text.values():
                print(value)
        # print("heloooooooooo",text[i].values())
        # Dividing Preorocessed text to small chunks
        if not flag:
            chunks = lmql_generator.chunk_text(text, size)
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i + 1}:")
                print(len(chunk))
                print("=" * 20)
        print("2000")
        
        print()
        
        print("model is ",model_llm)
        # calling LMQL query to generate MCQs
        mcqs = lmql_generator.query(chunks =text if flag else chunks, flag=flag,model=model_llm)
        print(mcqs)
        # Evaluating the generated MCQs using Chatgpt
        evaluate_mcqs = EvaluateMCQ()
        res=mcqs
        res_after_eval = []
        if flag:
            i=0
            for value in text.values():
                for j in range(int(num_q_each)):
                    MCQ = [res[i][j].variables['QUESTION'],res[i][j].variables['A'],res[i][j].variables['B'],res[i][j].variables['C'],res[i][j].variables['D'], res[i][j].variables['ANSWER']]
                    print("text is ", value)
                    print("MCQ is ", MCQ)
                    
                    response_eval = evaluate_mcqs.evaluate(document = value,MCQ= MCQ, flag=flag)
                    if response_eval.variables['VAL_QUE'] != 'Good' or response_eval.variables['VAL_OPT']!='Related':
                        res_after_eval.append(response_eval)
                        print(response_eval)
                        print(MCQ)
                    time.sleep(30)
                i+=1
        else:
            for i in range(len(chunks)):
                for j in range(int(num_q_each)):
                    MCQ = [res[i][j].variables['QUESTION'],res[i][j].variables['A'],res[i][j].variables['B'],res[i][j].variables['C'],res[i][j].variables['D'], res[i][j].variables['ANSWER']]
                    
                    response_eval = evaluate_mcqs(document = chunk[i],MCQ= MCQ, flag=flag)
                    if response_eval.variables['VAL_QUE'] != 'Good' or response_eval.variables['VAL_OPT']!='Related':
                        res_after_eval.append(response_eval)
                        print(response_eval)
                        print(MCQ)
                    time.sleep(30)

            
            # ans = lmql_generator.query(model = model_llm)
            # print("ans")

        # MCQ = res_after_eval.variables
        # res_after_eval = evaluate(document=chunks[0], MCQ=MCQ)

        # Making the generated MCQs into PDF 
        def generate_pdf(mcqs):
        # Convert MCQs to HTML
            html_content = "<h1>Generated MCQs</h1>"
            for mcq in mcqs:
                html_content += f"<p>{mcq}</p>"

            # Define PDF file path
            pdf_file_path = 'generated_mcqs.pdf'

            # Convert HTML to PDF
            pdfkit.from_string(html_content, pdf_file_path)

            return pdf_file_path
        
        # Send the PDF to user mail
        def send_mcqs_email(pdf_file_path):
            msg = Message('Generated MCQs PDF', recipients=[mail])  # Change to user's email address
            msg.body = 'Please find attached the generated MCQs PDF.'
            with app.open_resource(pdf_file_path) as pdf_file:
                msg.attach('generated_mcqs.pdf', 'application/pdf', pdf_file.read())
            mail.send(msg)
        
        mcq_path=generate_pdf(res_after_eval)
        send_mcqs_email(mcq_path)
        

        return jsonify({"message": "File uploaded successfully."})

    return render_template('index.html', form=form)


if __name__ == '__main__':
    text = '''
    'set', 'algorithm', 'machine', 'learning', 'example', 'cat', 'datum', 'dev', 'training', 'image', 'error', 'test', 'neural', 'learn', 'draft', 'page', 'network', 'team', 'ng', 'performance', 'distribution', 'yearning', 'andrew', 'picture', 'different', 'mobile', 'use', 'number', 'internet', 'human', 'suppose', 'variance', 'idea', 'train', 'new', 'big', 'size', 'help', 'small', 'build', 'category', 'large', 'task', 'high', 'system', 'huge', 'well', 'increase', 'classifier', 'bias', 'try', 'app', 'rate', 'give', 'level', 'work', 'label', 'city', 'feature', 'usually', 'need', 'model', 'include', 'look', 'add', 'curve', 'y', 'user', 'x', 'metric', 'dataset', 'know', 'parameter', 'improve', 'regularization', 'analysis', 'want', 'problem', 'technique', 'discuss', 'direction', 'reduce', 'effect', 'take', 'long', 'deep', 'book', 'define', 'evaluate', 'draw', 'progress', 'come', 'evaluation', 'e', 'accuracy', 'estimate', 'process', 'choose', 'doctor', 'care'
    '''
    # lmql_generator = GenerateMCQ()
    model_llm = initialize_model()

    # # Your main code here
    # mcqs = lmql_generator.query(chunks= text,model=model )
    # print(mcqs)
    app.run(debug=True, use_reloader=False)


