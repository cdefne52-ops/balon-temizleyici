import base64
import io
import os
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, abort

from balloon_core import clean_balloons, _EASYOCR_AVAILABLE
from languages import LANGUAGES, DEFAULT_LANG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

LANG_MAP = {k: (v[0], v[1]) for k, v in LANGUAGES.items()}


def _decode_upload(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _encode_preview(img, max_dim=900):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _safe_name(name):
    name = os.path.basename(name)
    return name or 'image.png'


@app.route('/')
def index():
    return render_template('index.html', easyocr_available=_EASYOCR_AVAILABLE,
                            languages=LANGUAGES, default_lang=DEFAULT_LANG)


@app.route('/sw.js')
def service_worker():
    resp = send_from_directory(os.path.join(BASE_DIR, 'static'), 'sw.js')
    resp.headers['Service-Worker-Allowed'] = '/'
    return resp


@app.route('/api/process', methods=['POST'])
def process():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'Resim bulunamadı'}), 400

    selected_langs = request.form.getlist('langs') or [DEFAULT_LANG]
    tess_langs = tuple(LANG_MAP.get(l, ('eng',))[0] for l in selected_langs)
    easy_langs = tuple(LANG_MAP.get(l, ('en',))[1] for l in selected_langs)
    engine = request.form.get('engine', 'auto')

    results = []
    for f in files:
        filename = _safe_name(f.filename)
        img = _decode_upload(f)
        if img is None:
            results.append({'filename': filename, 'error': 'Resim okunamadı'})
            continue

        langs = easy_langs if (engine == 'easyocr' or (engine == 'auto' and _EASYOCR_AVAILABLE)) else tess_langs
        try:
            cleaned = clean_balloons(img, langs=langs, engine=engine)
        except Exception as e:
            results.append({'filename': filename, 'error': str(e)})
            continue

        out_path = os.path.join(OUTPUT_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
            filename = filename + '.png'
            out_path = out_path + '.png'
        cv2.imwrite(out_path, cleaned)

        results.append({
            'filename': filename,
            'preview': _encode_preview(cleaned),
        })

    return jsonify({'results': results})


@app.route('/api/download/<path:filename>')
def download(filename):
    path = os.path.join(OUTPUT_DIR, _safe_name(filename))
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=_safe_name(filename))


@app.route('/api/download_all')
def download_all():
    names = request.args.getlist('f')
    if not names:
        abort(400)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            path = os.path.join(OUTPUT_DIR, _safe_name(name))
            if os.path.isfile(path):
                zf.write(path, _safe_name(name))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name='temizlenmis_balonlar.zip')


if __name__ == '__main__':
    print("EasyOCR available:", _EASYOCR_AVAILABLE)
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' if 'PORT' in os.environ else '127.0.0.1'
    app.run(host=host, port=port, debug=True)
