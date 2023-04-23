import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('RandomForest_Classifier.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get the user input from the form
    day = int(request.form['day'])
    month = (request.form['month'])
    education=request.form['education']
    balance = float(request.form['balance'])
    duration = int(request.form['duration'])
    age = int(request.form['age'])
    job = request.form['job']
    housing = request.form['housing']
    contact=request.form['contact']
    poutcome=request.form['poutcome']
    marital=request.form['marital']
    previous=int(request.form['previous'])
    pdays=int(request.form['pdays'])
    campaign=int(request.form['campaign'])
    
    dummycontact__telephone =0
    dummycontact__unknown=0 
    dummycontact__cellular=0

    if contact=='telephone':
        dummycontact__telephone =1
    elif contact=='cellular':
        dummycontact__cellular=1
    elif contact=='unknown':
        dummycontact__unknown=1
    dummypoutcome__other   =0  
    dummypoutcome__success  =0 
    dummypoutcome__unknown  =0
    dummypoutcome__failure =0  
    if poutcome=='success':
        dummypoutcome__success =1
    elif poutcome=='other':
        dummypoutcome__other=1
    elif poutcome=='failure':
        dummypoutcome__failure=1      
    elif poutcome=='unknown':
        dummypoutcome__unknown=1 
  
    data = pd.DataFrame({
        'month':[month],
        'education':[education],
        
        'job':[job],
        
        'housing':[housing],
       
        'marital_ordinal':[marital],
                 
    'dummycontact__unknown':[dummycontact__unknown],
    'dummypoutcome__success'  :[dummypoutcome__success],
    'dummypoutcome__unknown'  :[dummypoutcome__unknown]   
      })
      
    df = pd.DataFrame(data)
    month_dict={'apr':4,'may':5,'jan':1,'feb':2,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'mar':3,'dec':12}
    df['month']= df['month'].map(month_dict) 
    
    job_dict={'management': 2566,'blue-collar': 1944,'technician': 1823, 'admin.': 1334, 'services': 923, 'retired': 778, 'self-employed': 405, 
    'student': 360, 'unemployed': 357, 'entrepreneur': 328, 'housemaid': 274,'unknown': 70}
    df['job']=df['job'].map(job_dict)
    
    housing_dict={'yes':1,'no':0}
    df['housing']=df['housing'].map(housing_dict)
    
    education_dict={'secondary': 5476, 'tertiary': 3689, 'primary': 1500, 'unknown': 497}
    df['education']=df['education'].map(education_dict)
    
    marital_dict={'married': 0, 'divorced': 1, 'single': 2}
    df['marital_ordinal']=df['marital_ordinal'].map(marital_dict)
    #get dummies
   
     
    #scaled data
    data1=pd.DataFrame({
        'age':[age],
        'balance':[balance],
        
        'day':[day],'duration':[duration],
        'campaign':[campaign],'pdays':[pdays],
        'previous':[previous]
  
      })
    df1=scaler.transform(data1[ ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']])
       
    df2 = pd.DataFrame(df1, columns=['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])   
     
    
       
    df = pd.concat([df, df2], axis=1) 
    #df = df.drop(['feature_2', 'feature_3'], axis=1)   
 
    df=df[['duration', 'month', 'day', 'age', 'balance', 'campaign', 'job','dummypoutcome__success', 'dummycontact__unknown',
    'housing','education', 'dummypoutcome__unknown', 'marital_ordinal', 'pdays', 'previous']]

    # Make a prediction using the trained model
    print(df.columns)
    #prediction = model.predict(df)
    prediction = model.predict(df)[0]
    prediction = 'yes' if prediction == 1 else 'no'
    
    return render_template('index.html', prediction=prediction)
    
    

    # Return the prediction result to the HTML template
    #return render_template('index.html', prediction=prediction[0])
if __name__ == '__main__':
    app.run(debug=True)
  