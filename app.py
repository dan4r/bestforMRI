from flask import Flask, render_template, request, url_for, flash, redirect

from data_processing import predict, add_sample

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    np_core = request.args.get('np_core', default = '-', type = str)
    emic_x_size = request.args.get('emic_x_size', default = 1, type = float)
    emic_y_size = request.args.get('emic_y_size', default = 1, type = float)
    emic_z_size = request.args.get('emic_z_size', default = 1, type = float)
    print(np_core, emic_x_size, emic_y_size, emic_z_size)
    r1, r2 = '-', '-'
    if np_core != '-':
        r1, r2 = predict(np_core, emic_x_size, emic_y_size, emic_z_size)
    return render_template('index.html', r1=r1, r2=r2, formula=np_core)

@app.route('/add_sample', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        np_core = request.form['np_core']
        emic_x_size = request.form['emic_x_size']
        emic_y_size = request.form['emic_y_size']
        emic_z_size = request.form['emic_z_size']
        r1 = request.form['r1']
        r2 = request.form['r2']
        print(np_core, emic_x_size, r1, r2)
        add_sample(np_core, emic_x_size, emic_y_size, emic_z_size, r1, r2)
        return redirect('/add_sample')

    return render_template('add.html')
