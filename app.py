import pickle

from flask import Flask,render_template, request
from sklearn.preprocessing import StandardScaler
#model.pkl -trained ml model

#Deserialize - read the binary file-Ml model
clf=pickle.load(open('model.pkl','rb'))

###############################################################################################
#for getting range decided on xtrain-repeat the Ml steps till Normalization again
import pandas as pd
df=pd.read_csv("SUV_Purchase.csv")

#step-2
df=df.drop(['User ID','Gender'],axis =1)

#Loading the data
#setting the data into input and output values
X=df.iloc[:,:-1].values #iloc==>index location 2D array
Y=df.iloc[:,-1:].values #2D array

#Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

################################################################################################


app=Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(X_train)

    output=clf.predict(sst.transform([features]))
    print(output) #list format
    if(output[0]) == 0:
        return render_template('index.html',pred=f'The Person will not be able to purchase the SUV')
    else:
        return render_template('index.html',pred=f'The Person will be able to purchase the SUV')

if __name__ == "__main__":
    app.run(debug=True)