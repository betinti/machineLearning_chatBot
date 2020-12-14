# -*- coding: utf-8 -*-


import io
import random
import string 
import warnings
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#avisos de erros
import warnings
warnings.filterwarnings('ignore')
#fazer o download dos pacotes, apenas na primeira rodada
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')


"""
Grupo: Bernado Tinti e Leonardo Silva.
  Sobre o ChatBot: Chamamos nosso chatbot de "Turing", e ele tem como função responder questões relacionadas a "Machine Learning". O corpus usado é constituido de varios textos sobre o tema, as fontes desses textos estão mais abaixo em Bibliografia. O idioma
  utilizado deve ser o ingles.

  Perguntas Possiveis:
    - What does machine learning involves?
    - When was machine learning coined?
    - What is deep learning?
    - When the term machine learning was used for the first time?
    - Who invented machine learning?
    - What is the most commom application of machine learning?
    - How much a machine learning engineer expect a salary?
    - Can machine learning translate?
    - How Supervised machine learning works?
    - What is the difference between ML and AI?
    - What is the newest thing about machine learning?
"""




"""
Bibliografia:
  https://stacks.stanford.edu/file/druid:jt687kv7146/jt687kv7146.pdf
  https://en.wikipedia.org/wiki/Machine_learning
  https://www.ibm.com/cloud/learn/machine-learning
  https://www.nature.com/articles/nrg3920
  https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-applications
"""



# Processando o corpus no programa
machineLearn = "machineLearn.txt"
machineLearn_File = open(machineLearn, "r")
raw = machineLearn_File.read()
sent_tokens = nltk.sent_tokenize(raw) # lista de documentos

#lemarization e formação do dicionario em inglês
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


sem_pontuacao = dict((ord(punct), None) for punct in string.punctuation)  #retirar a pontução

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(sem_pontuacao)))


# "Humanizando" o bot

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey")    #cumprimento
GREETING_RESPONSES = ["hi", "hey", "hi :)", "hi there", "hello", ":P", ";)"]                   #resposta

#resposta randomica ao cumprimento do ususuário 
def greeting(sentence):
     for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)




def response(user_response):
    
    if(user_response == "who are you?" or user_response == "who are you"):
      return "Turing: I'm Alan Turing, I improved a lot that machine now you call computer"
    
    robo_response='Turing: '
    sent_tokens.append(user_response) #inclusão da pergunra 
   

    #procurando pela pergunta dentro do corpus

    #TF-IDF (em inglês)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort() #ordenação da matriz
    req_tfidf = flat[-2]
    if(req_tfidf==0):   #caso não ache nenhum relação com o texto
        robo_response=robo_response+"Sorry! I don't know what you asking for. :("
        sent_tokens.remove(user_response)
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        sent_tokens.remove(user_response)
        return robo_response

loop=True
print("Hello, I'm Turing, what's your name? ")
name = input(">>> ")
print()
print("Turing: Welcome " + name)
print("Turing: I will answer your questions about Machine Learning.") 
print("Turing: If you want to exit, type 'exit'!")
print()
while(loop==True):
    print()
    print("Turing: Make your questian ...")
    print(name + ": ", end=" ")
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='exit'):
        if(user_response=='thanks' or user_response=='thank you' or user_response=='tks' or user_response=='thx'):
            print("Turing: glad to help!")
        else:
            if(greeting(user_response)!=None):
                print("Turing: "+greeting(user_response))
            else:
                print(response(user_response))
    else:
        loop=False
        print("Turing: Always here! Bye!")


