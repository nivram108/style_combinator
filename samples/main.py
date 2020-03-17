import os
import sys
from flask import Flask, request, url_for, send_from_directory,render_template,redirect
from werkzeug import secure_filename
# need run.py
import run
# end need
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


styles = ["ff", "food_cube", "sun_set", "the_scream", "candy", "oil", "wave", "wind_valley", "none"]
deteceted_objects = ["background"]
uploaded = 0
transfered = 0

# need run.py

# run.setup()

# end need

@app.route('/transfer', methods=['POST'])
def transfer_file():
    global transfered
    global deteceted_objects
    class_style_dict = {"background" : "none"}
    for obj in deteceted_objects:
        choosed_style = request.form.get(obj)
        class_style_dict[obj] = choosed_style
    #need run.py
    run.demo.set_style(class_style_dict)
    run.generate()
    # end need
    transfered = 1
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded
    global transfered
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploaded = 1
            transfered = 0
            filename = secure_filename(file.filename)
            file.save("static/input.png")
            # need run.py
            run.setup()
            run.set_input_image("static/input.png")
            run.set_output_path("static/result.png")
            print("set output")
            run.classify()
            global deteceted_objects
            deteceted_objects = run.get_contained_class()
            # end need
    return redirect(url_for('index'))

@app.route('/', methods=['GET'])
def index():
    print("uploaded",uploaded)
    print("transfered",transfered)
    return render_template('index.html',styles=styles,deteceted_objects=deteceted_objects,uploaded=uploaded,transfered=transfered)



if __name__ == '__main__':
    app.run(port=5550,host="0.0.0.0")
