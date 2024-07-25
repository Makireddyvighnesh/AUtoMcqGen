import lmql
import os
import pandas as pd
from tabula import read_pdf
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
import time

openai_api_key = os.getenv("OPENAI_API_KEY")
import nest_asyncio
nest_asyncio.apply()
from rouge import Rouge
rouge=Rouge()
import textwrap
    
model = lmql.model("local:llama.cpp:zephyr-7b-beta.Q4_K_M.gguf", 
                        tokenizer = 'HuggingFaceH4/zephyr-7b-beta', inprocess = True) 

class GenerateMCQ():
    def __init__(self,n):
        self.n = n
        self.result =[]
    
    @lmql.query
    def generate_questions(self,document):
        '''lmql
        sample(temperature=1.0, n=int(self.n))  # decoder 
        print(0)
        
        "Q:Look at the given document {document}. Your task is to generate a multiple-choice question which has only 4 options based on the information provided in the document. "
        print(1)
        "Please ensure that the question is clear and avoid directly referencing the document in the question itself. "
        "Additionally, make the options related and slightly tricky."
        # for i in range(10):
        "Question: [QUESTION]" # where stops_at(QUESTION,'?') and stops_before(QUESTION, 'A:') and not "summarise" in QUESTION and len(TOKENS(QUESTION))<70 and \
        #                not 'paraphrased' in QUESTION and not 'essay' in QUESTION and not "summary" in QUESTION and \
        #                not 'Write' in QUESTION and not '----' in QUESTION and len(QUESTION)>10 
        print(f"Question:\n {QUESTION}") 
        "The options for the question {QUESTION}  "
        time.sleep(30)
        "So the first option is: "
        # for i in range(100):
        "A: [A]" #where stops_before(A, 'B') and len(TOKENS(A))<50 and not 'Generate according to:' in A and len(TOKENS(A))>2 and stops_before(A, "So the second option is: ")
        time.sleep(50)
        #  stops_at(A,'.') and 
        print(f"A: {A}")
        
        "B: [B]" # where stops_before(B,'C') and len(TOKENS(B))<50 and not 'Generate according to:' in B and len(TOKENS(B))>2 and not '2.' in B and stops_before(B, "So the third option is: ")
        print(f"B: {B}")
        time.sleep(40)
        # "So the third option is: "
        "C: [C]" # where stops_before(C,'D') and len(TOKENS(C))<50 and not 'Generate according to:' in C and len(TOKENS(C))>2 and not '3.' in C and stops_before(C, "So the forth option is: ")
        print(f"C :\n {C}")
        time.sleep(50)
        # "So the forth option is: "
        "D: [D]" # where stops_before(D, '\nThe answer for this question is ') and  len(TOKENS(D))<50  and stops_at(D, '.') and not 'Generate according to:' in D \
                       # and not 'E:' in D and len(TOKENS(D))>2  and not '4.' in D 
        print(f"D :\n {D}")
        # for i in range(5):
        "And the answer for the question is "
        "Answer: [ANSWER] \n" where ANSWER in ["A","B","C","D"]
        print(f"ANSWER: {ANSWER}")
        time.sleep(50)
        "Reason: [REASONING]" where len(TOKENS(REASONING))>30 and stops_at(REASONING, '\n')
        print(f"Reason: {REASONING}")
        
        '''

    @lmql.query
    def generate_blanks(self,document):
        '''
        sample(temperature = 1.0, n=self.n)
        print(1)
        # print(document)
        
        "Your task is to generate blanks for the given document {document}. Ensure that the blanks are clear to the end user who is writing an exam for your questions."
        "The document provided to you is not given to the user to answer the questions. Therefore, avoid using phrases like 'given in the document', 'as from the document', and similar."
        "If a particular question requires any additional information or snippet to answer the blanks, provide it; otherwise, leave it empty."
        # "Keep the rules while generating blanks.:"
        # "Before generating blanks have a look at the pattern of how blanks would be"
        # "size of integer in bytes is:____"
        # "Answer is: 16bits"
        # "True or False: Void function return values"
        # "Answer if False"
        "Now that you have some understanding of how blanks should look and how clear they should be, let's generate the blanks:"
        "Q: [BLANK]" where len(TOKENS(BLANK))<70 and stops_before(BLANK,'Answer to the question is ')
        print(f"BLANK: {BLANK}")
        # time.sleep(50)
        "Answer to the question is "
        "A:[ANSWER]" where len(TOKENS(ANSWER))<15
        print(f"ANSWER: {ANSWER}")
        # time.sleep(50)
        "Explanation to the question: "
        "[REASON]" where len(TOKENS(REASON))<70 and stops_at(REASON, '.')
        print(f"Reason: {REASON}")
        # time.sleep(30)
        '''

    @lmql.query
    def generate_LAQ(self,document):
        '''
        sample(temperature = 1.0, n=self.n)
        "Your task is to generate Long Answer Questions for the given document {document}. Ensure that the questions are clear to the end user who is writing an exam for your questions."
        "The document provided to you is not given to the user to answer the questions. Therefore, avoid using phrases like 'given in the document', 'as from the document', and similar."
        "If a particular question requires any code snippet to answer the question, provide it; otherwise, leave it empty."
        "Keep the following rules in mind while generating questions:"
        "Let's now generate the question: "
        "[LAQ]" where len(TOKENS(LAQ))<70 and stops_at(LAQ,'?')
        "The answer to the question is: "
        "[ANSWER]" where len(TOKENS(ANSWER))>200 and stops_at(ANSWER,'.')
        '''

    def chunk_text(self,text, size):
        chunks = textwrap.wrap(text, size)
        return chunks

    # @lmql.query
    def query(self, chunks,flag,model):
        # print(type(chunks))
        # print(chunks)
        print(flag)
        if not flag:
            for i, chunk in enumerate(chunks):
                if i==3: break
                    
                print(i)
                
                print(model)
                res = self.generate_questions(document=chunks[i], model=model)
                self.result.append(res)
        else:
            
            print(flag)
            
            
            res = self.generate_questions(document=chunks, model=model)
            # print(res[0].variables)
            # print(res[1].variables)
            self.result.append(res)

        return self.result

class EvaluateMCQ():
    def __init__(self):
        self.result=[]
    
    @lmql.query
    def evaluate(self,document, MCQ):
      '''lmql
      print("Evaluating MCQs")
      "Your Task is to evaluate the MCQ question which is given to you later    using the reference document {document}"
      q, A, B,C,D, answer = MCQ
      # print(q,a,b,c,d)
      "Look at the "
      "Question: {q} and Options "
      "A: {A}"
      "B: {B}"
      "C: {C}"
      "D: {D}"
      "So let's evaluate the question and options proided before and Try to findout any inaccuaracies. "
      "So let's check whether given quetstion is related to the document or not [VAL_QUE] " where VAL_QUE in ["Good", "not related"]
      print(f"VAL_QUE: {VAL_QUE}")
      time.sleep(20)
      "Now lets check out the options whether they are related to the given question or not and options should not be empty . [VAL_OPT]" where VAL_OPT in ["Related", "some options are empty", "Not Realated"]
      print(f"VAL_OPT: {VAL_OPT}")
      "Now lets check whether the given answer {answer} is correct or not [ANS] " where ANS in ["Correct", "Incorrect"]
      
      if VAL_QUE != 'Good' or VAL_OPT != 'Related':
        time.sleep(30)
      if VAL_QUE!='Good':
        "The modified question is :[AF_MOD_QUE]" where stops_at(AF_MOD_QUE,'?') and stops_before(AF_MOD_QUE, 'A') and not "summarize" in AF_MOD_QUE and len(TOKENS(AF_MOD_QUE))<70 
        print(f"AF_MOD_QUE :{AF_MOD_QUE}")
        time.sleep(60)
      if VAL_OPT!= "Related":
        "The modified options are :"
        "A: [A]" where stops_before(A, 'B') and len(TOKENS(A))>2 and not 'Generate according to:' in A
        #  stops_at(A,'.') and 
        # print(f"A: {A}")
        time.sleep(50)
        "B: [B]" where stops_before(B,'C') and len(TOKENS(B))>2 and not 'Generate according to:' in B
        # print(f"B: {B}")
        time.sleep(20)
        "C: [C]" where stops_before(C,'D') and len(TOKENS(C))>2 and not 'Generate according to:' in C
        # print(f"C :\n {C}")
        time.sleep(40)
        "D: [D]" where stops_before(D, '\nThe answer for this question is ') and  len(TOKENS(D))>2  and stops_at(D, '.') and not 'Generate according to:' in D \
                       and not 'E:' in D
        if ANS != "Correct":
          time.sleep(40)
          "Correct Answer is: [ANSWER]" where ANSWER in ["A", "B","C","D"]
        '''
