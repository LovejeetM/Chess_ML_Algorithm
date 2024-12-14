from flask import Flask, render_template, request, jsonify
from ml import process_chess_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files or 'turn' not in request.form:
        return jsonify({'error': 'Missing image or turn input'}), 400

    file = request.files['file']
    turn = request.form['turn']

    if file and turn:
        fen = process_chess_image(file, turn)
        return jsonify({'fen': fen})
    else:
        return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(debug=True)
