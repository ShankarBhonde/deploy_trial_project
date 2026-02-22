#from flask import Flask, render_template, request,redirect,url_for
from flask import Flask, render_template, request, redirect, url_for, flash
#from flask_mysqldb import MySQL
from ultralytics import YOLO

import joblib
import numpy as np

app = Flask(__name__)
#--------------------------------------------
#load plant disease model 
predict_plant_disease = YOLO("best.pt")

#UPLOAD_FOLDER = "static/uploads"
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#-------------------------------------------------------------
# Load models
crop_model=joblib.load("crop.pkl")
#with open("crop.pkl", "rb") as f:
#    crop_model = pickle.load(f)

soil_model=joblib.load('soil.pkl')
#with open("model_soil.pkl", "rb") as f:
#    soil_model = pickle.load(f)

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/index", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/crop", methods=["GET", "POST"])
def crop():
    prediction = None
    if request.method == "POST":
        features = np.array([[
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]])
        prediction = crop_model.predict(features)[0]


    return render_template("crop.html", prediction=prediction)

@app.route("/soil", methods=["GET", "POST"])
def soil():
    prediction = None
    if request.method == "POST":
        features = np.array([[
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["ph"])
        ]])
        prediction = soil_model.predict(features)[0]

    return render_template("soil.html", prediction=prediction)




'''
#mysql connection 
app.secret_key="secrete123"
#mysql configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Shankar'
app.config['MYSQL_DB'] = 'feedback'

mysql= MySQL(app)

#feedback form 
@app.route('/user_feedback',methods=['POST'])
def feedback():
    if request.method=='POST':
        name=request.form.get('name')
        mo_no=request.form.get('mo_no')
        message=request.form.get('message')

        cur=mysql.connection.cursor()
        cur.execute(
            "Insert into farmer_details(name,mo_no,message) values(%s , %s, %s)",(name,mo_no,message)

        )
        mysql.connection.commit() 
        cur.close()
        flash("Feedback submitted successfully! 🌾", "success")

        #return redirect(url_for('feedback'))
    return render_template('feedback.html',messages=messages)
    #return redirect(url_for('index'))
    

    #return render_template("feedback.html")
    return " seccussfull Done , pls Go Back"


'''

#---------------------------------------------------------------------------
#plant disease detection 

@app.route("/plant_disease")
def plant_disease():
    return render_template("plant_disease.html")

#1. local file upload
@app.route("/plant_upload", methods=["POST"])
def plant_upload():
    file = request.files.get("image")
    if not file:
        return redirect(url_for("plant_disease"))

    image_path = os.path.join("static/uploads", file.filename)
    file.save(image_path)

    prediction, confidence = predict_plant_disease(image_path)

    return render_template(
        "plant_disease.html",
        image_path=image_path,
        prediction=prediction,
        confidence=confidence
    )


#2. camera route
@app.route("/plant_camera", methods=["POST"])
def plant_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return redirect(url_for("plant_disease"))

    image_path = "static/uploads/camera.jpg"
    cv2.imwrite(image_path, frame)

    prediction, confidence = predict_plant_disease(image_path)

    return render_template(
        "plant_disease.html",
        image_path=image_path,
        prediction=prediction,
        confidence=confidence
    )


#------------------------------------------------------------



if __name__ == "__main__":
    app.run(debug=True)





