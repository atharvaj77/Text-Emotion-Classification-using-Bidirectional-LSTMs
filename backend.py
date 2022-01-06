from flask import Flask,render_template,session,url_for,redirect
from flask_wtf import FlaskForm
from six import print_
from wtforms import TextField, SubmitField
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd



def get_words(tokenizer,model,data):

    classes = ["sadness","anger","love","surprise","fear","joy"]

    input_text = data['text'].strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text],maxlen=60, truncating='pre')
    predictions = model.predict(pad_encoded)

    pred_class = classes[np.argmax(predictions)]

    print(predictions)

    return pred_class

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


class SearchBar(FlaskForm):
    
    text = TextField('Search Bar')
    submit = SubmitField('Get Emotion')


@app.route('/',methods=['GET','POST'])
def index():
 search = SearchBar()
 
 if search.validate_on_submit():
     
     session['text'] = search.text.data
     
     return redirect(url_for('prediction'))
 
 return render_template('home.html',form=search)
     


user_model = load_model('model_lstm.h5')

with open('tokenizer.pickle','rb') as handle:
    user_tokenizer = pickle.load(handle)
    
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    
    content = {}
    
    content['text'] = session['text']
    results = get_words(user_tokenizer,user_model,content)
    
    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run()