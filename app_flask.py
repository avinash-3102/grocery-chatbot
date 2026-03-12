from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# load model (SavedModel format)
model = tf.saved_model.load("grocery_saved_model")
infer = model.signatures["serving_default"]

# correct class order (same as training)
class_names = [
"beans",
"black_beans",
"bread",
"coffee",
"flour",
"milk",
"oil",
"rice",
"sugar"
]

# load grocery database
with open("grocery_database.json") as f:
    grocery_db = json.load(f)


def predict_image(image):

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0).astype(np.float32)

    preds = infer(tf.constant(img))

    # get prediction tensor
    preds = list(preds.values())[0].numpy()

    index = np.argmax(preds[0])
    confidence = float(preds[0][index])
    label = class_names[index]

    print("Predictions:", preds)
    print("Predicted:", label)

    return label, confidence

HTML = """
<!DOCTYPE html>
<html>

<head>

<title>Grocery AI Chatbot</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">

<style>

*{
box-sizing:border-box;
font-family:'Poppins',sans-serif;
}

body{
margin:0;
height:100vh;
display:flex;
justify-content:center;
align-items:center;
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.container{

width:550px;
padding:40px;
border-radius:20px;

background:rgba(255,255,255,0.1);
backdrop-filter:blur(10px);

box-shadow:0 20px 50px rgba(0,0,0,0.4);

text-align:center;

}

h2{
margin-bottom:25px;
font-weight:600;
}

.upload-box{

border:2px dashed rgba(255,255,255,0.5);
padding:30px;
border-radius:15px;
margin-bottom:20px;

cursor:pointer;

transition:0.3s;
}

.upload-box:hover{
background:rgba(255,255,255,0.1);
}

input[type=file]{
display:none;
}

.upload-text{
font-size:16px;
}

#uploadStatus{
margin-top:10px;
font-size:14px;
color:#00ff88;
display:none;
}

input[type=text]{

width:100%;
padding:14px 18px;

border-radius:12px;

border:2px solid rgba(255,255,255,0.3);

background:white;

font-size:15px;

outline:none;

margin-bottom:15px;
}

button{

padding:14px 40px;

border:none;
border-radius:30px;

font-size:16px;
font-weight:500;

background:linear-gradient(45deg,#4facfe,#00f2fe);

color:white;
cursor:pointer;

transition:0.3s;
}

button:hover{

transform:scale(1.08);
box-shadow:0 10px 25px rgba(0,0,0,0.4);
}

.result{

margin-top:25px;
padding:20px;

border-radius:15px;

background:rgba(255,255,255,0.15);

border:1px solid rgba(255,255,255,0.3);

line-height:1.6;
}

</style>

</head>

<body>

<div class="container">

<h2>🛒 Grocery Conversational Chatbot</h2>

<form method="POST" enctype="multipart/form-data">

<label class="upload-box">

<div class="upload-text">📂 Click here to upload grocery image</div>

<input type="file" name="file" id="fileInput" required>

</label>

<div id="uploadStatus">✅ Image uploaded successfully</div>

<input type="text" name="question" placeholder="Ask something about the product">

<button type="submit">Analyze Product</button>

</form>

{% if result %}

<div class="result">

{{result|safe}}

</div>

{% endif %}

</div>

<script>

document.getElementById("fileInput").addEventListener("change", function(){

if(this.files.length > 0){

document.getElementById("uploadStatus").style.display="block";

}

});

</script>

</body>
</html>
"""


@app.route("/", methods=["GET","POST"])
def home():

    result=None

    if request.method=="POST":

        file=request.files["file"]
        question=request.form["question"]

        image=Image.open(file).convert("RGB")

        product,confidence=predict_image(image)

        info=grocery_db.get(product,{})

        description=info.get("description","No description available")
        nutrition=info.get("nutrition","Unknown nutrition")

        result=f"""
Detected Product: {product}<br>
Confidence: {confidence:.2f}<br><br>

Description: {description}<br>
Nutrition: {nutrition}<br><br>

User Question: {question}
"""

    return render_template_string(HTML,result=result)


if __name__=="__main__":
    app.run(debug=True)