from flask import Flask, request, send_from_directory, jsonify, send_file
import os
import base64
import json
import jwt
from functools import wraps
from datetime import datetime
import logging
import io

app = Flask(__name__)
app.config.update(
    IMAGE_FOLDER='static/images',
    ANNOTATION_FOLDER='annotations',
    FEEDBACK_FOLDER='feedback',
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', 'default-key'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)

# Create top-level folders if missing
for folder in [
    app.config['IMAGE_FOLDER'],
    app.config['ANNOTATION_FOLDER'],
    app.config['FEEDBACK_FOLDER']
]:
    os.makedirs(folder, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portal.log'),
        logging.StreamHandler()
    ]
)

def require_api_key(f):
    """Decorator to ensure all requests include a valid JWT in 'Authorization: Bearer <token>'."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No API key provided'}), 401
        try:
            token = auth_header.split(' ')[1]
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid API key'}), 401
        except Exception as e:
            logging.error(f"Authorization error: {str(e)}")
            return jsonify({'error': 'Authorization error'}), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    """Serve an index.html if desired."""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        logging.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'Error serving index page'}), 500

@app.route('/images')
def get_images():
    """
    Return a JSON list of .png files under static/images (recursively).
    E.g., ["NJ05_file1.png", "subdir/NJ05_file2.png", ...].
    """
    try:
        images = []
        for root, _, files in os.walk(app.config['IMAGE_FOLDER']):
            for f in files:
                if f.lower().endswith('.png'):
                    relpath = os.path.relpath(root, app.config['IMAGE_FOLDER'])
                    if relpath == '.':
                        images.append(f)
                    else:
                        images.append(os.path.join(relpath, f))
        images.sort()
        logging.info(f"Returning {len(images)} images")
        return jsonify(images)
    except Exception as e:
        logging.error(f"Error getting images list: {str(e)}")
        return jsonify({'error': 'Error retrieving images'}), 500

@app.route('/api/upload/<grid_square>', methods=['POST'])
@require_api_key
def upload_batch(grid_square):
    """
    Upload multiple .png files to static/images/, prefixing each with <grid_square>_ if needed.
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'Empty files list'}), 400

        logging.info(f"Processing {len(files)} files for upload to {grid_square}")
        saved_files = []

        for file in files:
            if file and file.filename:
                # Force the grid_square prefix if missing
                if not file.filename.startswith(grid_square):
                    filename = f"{grid_square}_{file.filename}"
                else:
                    filename = file.filename

                filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filename)

        logging.info(f"Uploaded {len(saved_files)} files for {grid_square}")
        return jsonify({
            'status': 'success',
            'uploaded': saved_files
        })
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    try:
        data = request.get_json()
        image_name = data.get('image_name')
        user_hash = data.get('user_hash')
        mask_data_raw = data.get('mask')

        # Extract mask image
        mask_data = mask_data_raw.split(',')[1]
        mask_image = base64.b64decode(mask_data)

        # Open and verify dimensions
        from PIL import Image
        import io

        # Get dimensions of original image
        image_path = os.path.join(app.config['IMAGE_FOLDER'], image_name)
        if os.path.exists(image_path):
            orig_img = Image.open(image_path)
            orig_width, orig_height = orig_img.size

            # Check mask dimensions
            mask_img = Image.open(io.BytesIO(mask_image))
            mask_width, mask_height = mask_img.size

            # If dimensions don't match, reject
            if mask_width != orig_width or mask_height != orig_height:
                return jsonify({
                    'status': 'error',
                    'message': f'Mask dimensions ({mask_width}x{mask_height}) do not match original image ({orig_width}x{orig_height})'
                }), 400

        # Save the mask if validation passes
        mask_filename = f"{os.path.splitext(image_name)[0]}_{user_hash}_mask.png"
        mask_path = os.path.join(app.config['ANNOTATION_FOLDER'], mask_filename)

        with open(mask_path, 'wb') as f:
            f.write(mask_image)

        return jsonify({'status': 'success'})

    except Exception as e:
        logging.error(f"Annotation save error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/annotations/<grid_square>', methods=['GET'])
@require_api_key
def get_annotations(grid_square):
    """
    Return all annotation .png files in the root folder that start with <grid_square>_.
    Each file's binary content is placed in the JSON response, which can be large.

    E.g.:
    {
      "grid_square": "NJ05",
      "annotations": {
        "NJ05_file1_mask.png": <raw-bytes>,
        "NJ05_file2_mask.png": <raw-bytes>
      }
    }

    Potential for large responses if many big .png exist.  Use caution.
    """
    try:
        annotations = {}
        root_dir = app.config['ANNOTATION_FOLDER']

        if not os.path.exists(root_dir):
            logging.warning(f"Annotations directory not found: {root_dir}")
            return jsonify({
                'grid_square': grid_square,
                'annotations': {}
            })

        matching_files = 0
        # Find all files matching the prefix "<grid_square>_" and suffix ".png"
        for filename in os.listdir(root_dir):
            if filename.startswith(grid_square + "_") and filename.endswith(".png"):
                matching_files += 1
                filepath = os.path.join(root_dir, filename)
                with open(filepath, 'rb') as f:
                    # Return binary data directly in the JSON
                    # (the pipeline writes it out raw or as text if 'isinstance(content, str)')
                    annotations[filename] = f.read()

        logging.info(f"Returning {matching_files} annotations for {grid_square}")
        return jsonify({
            'grid_square': grid_square,
            'annotations': annotations
        })
    except Exception as e:
        logging.error(f"Error fetching annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations-file/<filename>', methods=['GET'])
@require_api_key
def get_annotation_file(filename):
    """
    Return a single annotation file by filename.
    Returns the file as base64-encoded data in a JSON response.
    """
    try:
        root_dir = app.config['ANNOTATION_FOLDER']
        filepath = os.path.join(root_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Read the file and base64-encode
        with open(filepath, 'rb') as f:
            content = f.read()

        # Return as JSON with base64-encoded data
        return jsonify({
            "filename": filename,
            "data": base64.b64encode(content).decode('utf-8')
        })
    except Exception as e:
        logging.error(f"Error retrieving annotation file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/file/<filename>', methods=['GET'])
@require_api_key
def get_raw_file(filename):
    """
    Stream the file directly as a response with proper Content-Type.
    Can be used for both images and annotation masks.
    """
    try:
        # Check annotations folder first
        filepath = os.path.join(app.config['ANNOTATION_FOLDER'], filename)
        if not os.path.exists(filepath):
            # If not in annotations, check images folder
            filepath = os.path.join(app.config['IMAGE_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({"error": "File not found"}), 404

        # Determine MIME type based on extension
        mime_type = 'image/png' if filename.lower().endswith('.png') else 'application/octet-stream'

        # Stream file directly
        return send_file(filepath, mimetype=mime_type)
    except Exception as e:
        logging.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<grid_square>', methods=['GET'])
@require_api_key
def get_annotation_status(grid_square):
    """
    Report on annotation status for a grid square.
    Counts how many grid_square prefixed images exist and how many matching annotations exist.
    """
    try:
        # For images, check both root images folder and potential grid subfolder
        image_dir = app.config['IMAGE_FOLDER']
        grid_image_dir = os.path.join(app.config['IMAGE_FOLDER'], grid_square)

        # For annotations, check root annotations folder
        annotation_dir = app.config['ANNOTATION_FOLDER']

        status = {
            'grid_square': grid_square,
            'total_images': 0,
            'annotated_images': 0,
            'completed': False
        }

        # Count images in root folder with grid_square prefix
        if os.path.exists(image_dir):
            status['total_images'] += len([
                f for f in os.listdir(image_dir)
                if f.lower().endswith('.png') and f.startswith(f"{grid_square}_")
            ])

        # Count images in grid-specific subfolder if it exists
        if os.path.exists(grid_image_dir):
            status['total_images'] += len([
                f for f in os.listdir(grid_image_dir)
                if f.lower().endswith('.png')
            ])

        # Count annotations with matching grid_square prefix
        if os.path.exists(annotation_dir):
            status['annotated_images'] = len([
                f for f in os.listdir(annotation_dir)
                if f.endswith('_mask.png') and f.startswith(f"{grid_square}_")
            ])

        status['completed'] = (
            status['total_images'] > 0 and
            status['annotated_images'] >= status['total_images']
        )

        logging.info(f"Status for {grid_square}: {status['total_images']} images, {status['annotated_images']} annotations")
        return jsonify(status)
    except Exception as e:
        logging.error(f"Status check error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/<grid_square>', methods=['POST'])
@require_api_key
def submit_feedback(grid_square):
    """
    Receives feedback JSON and saves it under feedback/<grid_square>/feedback_YYYYmmdd-HHMMSS.json
    """
    try:
        feedback = request.get_json()
        if not feedback:
            return jsonify({'error': 'No feedback provided'}), 400

        feedback_dir = os.path.join(app.config['FEEDBACK_FOLDER'], grid_square)
        os.makedirs(feedback_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f'feedback_{timestamp}.json'
        filepath = os.path.join(feedback_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(feedback, f, indent=2)

        logging.info(f"Saved feedback for {grid_square}: {filename}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Feedback submission error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from the 'static' folder."""
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logging.error(f"Error serving static file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logging.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/delete/<grid_square>/<filename>', methods=['DELETE'])
@require_api_key
def delete_annotation(grid_square, filename):
    """
    Delete a specific annotation file.

    Ensures the filename matches the grid_square pattern for safety.
    Returns 404 if the file doesn't exist.
    """
    try:
        # Security check - make sure filename starts with grid_square
        if not filename.startswith(f"{grid_square}_"):
            return jsonify({'error': f"Filename does not match grid square {grid_square}"}), 400

        # Check that file exists
        filepath = os.path.join(app.config['ANNOTATION_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # Delete the file
        os.remove(filepath)
        logging.info(f"Deleted annotation file: {filename}")

        return jsonify({
            'status': 'success',
            'message': f"Successfully deleted {filename}"
        })

    except Exception as e:
        logging.error(f"Error deleting annotation {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-batch/<grid_square>', methods=['POST'])
@require_api_key
def delete_batch_annotations(grid_square):
    """
    Delete multiple annotation files in a single request.

    Expects a JSON body with a 'filenames' array.
    Returns lists of successfully deleted files and failures.
    """
    try:
        data = request.get_json()
        if not data or 'filenames' not in data:
            return jsonify({'error': 'No filenames provided'}), 400

        filenames = data['filenames']
        if not isinstance(filenames, list):
            return jsonify({'error': 'Filenames must be a list'}), 400

        deleted = []
        failed = []

        for filename in filenames:
            # Security check - ensure filename matches grid square
            if not filename.startswith(f"{grid_square}_"):
                failed.append({"filename": filename, "reason": "Filename does not match grid square"})
                continue

            filepath = os.path.join(app.config['ANNOTATION_FOLDER'], filename)
            if not os.path.exists(filepath):
                failed.append({"filename": filename, "reason": "File not found"})
                continue

            try:
                os.remove(filepath)
                deleted.append(filename)
                logging.info(f"Deleted annotation file: {filename}")
            except Exception as e:
                failed.append({"filename": filename, "reason": str(e)})
                logging.error(f"Error deleting {filename}: {str(e)}")

        return jsonify({
            'status': 'success',
            'deleted': deleted,
            'failed': failed,
            'message': f"Deleted {len(deleted)} files, {len(failed)} failed"
        })

    except Exception as e:
        logging.error(f"Error in batch deletion: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
