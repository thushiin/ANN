from flask import Flask,render_template,request
from keras.models import load_model
from PIL import Image
import numpy as np

data=load_model('cifar10model.h5')

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('cifar10.html')

@app.route('/predict', methods=['post'])
def pred():
    classes = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
    
    img=request.files['image']
    img=Image.open(img)
    img=img.resize((32,32))
    img=np.array(img)/255
    im=np.expand_dims(img,axis=0)
    m=data.predict(im)
    output=np.argmax(m)
    pred=classes[output]
    return render_template('cifar10.html',prediction=pred)

if __name__=='__main__':
    app.run(debug=True)