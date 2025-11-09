from flask import Flask, request, jsonify, render_template
import model_service # Import the analysis logic
import io

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the updated index.html from the 'templates' folder
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Handle File Upload
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    # Read the file content as a stream/bytes
    file_stream = file.read()

    # 2. Handle FSM Parameters
    try:
        base_thr = float(request.form.get('base_thr'))
        base_k = int(request.form.get('base_k'))
        base_m = int(request.form.get('base_m'))
        window = int(request.form.get('window'))
    except Exception:
        return jsonify({"error": "Invalid FSM parameters. Please check inputs."}), 400

    # 3. Run Analysis
    try:
        # Call the analysis logic from model_service.py
        results = model_service.analyze_data(file_stream, base_thr, base_k, base_m, window)
        return jsonify(results)
    
    except Exception as e:
        # Catch any errors (like FileNotFoundError, ValueError from data prep) and return them
        print(f"Analysis Failed with Python Error: {e}")
        return jsonify({"error": f"Backend Analysis Failed. Check server log for traceback. Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Flask will now start and also load the models in model_service.py
    app.run(debug=True)