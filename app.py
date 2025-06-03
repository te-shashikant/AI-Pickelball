from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import os
import uuid
import json
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
from pose_detection import process_video
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
from reportlab.lib.utils import ImageReader
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


def generate_pdf_report(feedbacks, score, accepted_segments, rejected_segments, pdf_path="analysis_report.pdf"):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    def write_line(text, font_size=12, color=colors.black, indent=0):
        nonlocal y
        c.setFont("Helvetica", font_size)
        c.setFillColor(color)
        c.drawString(margin + indent, y, text)
        y -= font_size + 6

    def draw_segment_image(image_path, x, y, max_width=200, max_height=150):
        try:
            img = ImageReader(image_path)
            iw, ih = img.getSize()
            aspect = ih / iw
            if iw > max_width:
                iw = max_width
                ih = iw * aspect
            if ih > max_height:
                ih = max_height
                iw = ih / aspect
            c.drawImage(img, x, y - ih, width=iw, height=ih)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    c.setTitle("Pickleball Pose Analysis Report")
    write_line("Pickleball Pose Analysis Report", font_size=18, color=colors.darkblue)
    write_line(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", font_size=10)
    write_line("-" * 90)

    write_line(f"Final Score: {score}", font_size=14, color=colors.green if score >= 75 else colors.red)
    write_line("")

    write_line("🔍 Feedback Summary:", font_size=14, color=colors.darkblue)
    for fb in feedbacks:
        write_line(f"• {fb}", font_size=12, indent=10)
    write_line("")

    # Accepted segments
    write_line(f"✅ Accepted Segments: {len(accepted_segments)}", font_size=13, color=colors.green)
    x_img = margin + 20
    for seg in accepted_segments:
        write_line(f" - From {seg['start']:.2f}s to {seg['end']:.2f}s", font_size=11, indent=10)
        if "thumbnail" in seg:
            local_path = seg["thumbnail"].lstrip("/").replace("/", os.sep)
            draw_segment_image(local_path, x_img, y)
            y -= 160
    write_line("")

    # Rejected segments
    write_line(f"❌ Rejected Segments: {len(rejected_segments)}", font_size=13, color=colors.red)
    for seg in rejected_segments:
        write_line(f" - From {seg['start']:.2f}s to {seg['end']:.2f}s", font_size=11, indent=10)
        write_line(f"   Reasons: {', '.join(seg['reasons'])}", font_size=10, indent=20)
        if "thumbnail" in seg:
            local_path = seg["thumbnail"].lstrip("/").replace("/", os.sep)
            draw_segment_image(local_path, x_img, y)
            y -= 160

    c.showPage()
    c.save()


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GIF_FOLDER'] = 'static/gifs'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GIF_FOLDER'], exist_ok=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String, unique=True, nullable=False)
    score = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video = request.files.get('video')
        mode = request.form.get('mode', 'stroke')

        if not video:
            return render_template('upload.html', error="No video uploaded.")

        filename = secure_filename(video.filename)
        video_id = str(uuid.uuid4())[:8]
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
        video.save(video_path)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_processed_{filename}")

        feedbacks, final_score, accepted_segments, rejected_segments = process_video(video_path, output_path, mode)

        gif_output_folder = os.path.join(app.config['GIF_FOLDER'], video_id)
        os.makedirs(gif_output_folder, exist_ok=True)

        segment_data = []
        segment_id = 1

        def save_frames_and_gif(segment, label):
            nonlocal segment_id
            frames = segment["frames"]
            gif_filename = f"{label}_{segment_id}.gif"
            gif_path = os.path.join(gif_output_folder, gif_filename)
            frame_urls = []
            thumbnail_url = None

            images = [cv2.cvtColor(f["frame"], cv2.COLOR_BGR2RGB) for f in frames]
            if images:
                pil_imgs = [Image.fromarray(img) for img in images]
                pil_imgs[0].save(gif_path, save_all=True, append_images=pil_imgs[1:], duration=100, loop=0)

                for idx, img in enumerate(pil_imgs):
                    frame_filename = f"{label}_{segment_id}_frame{idx+1}.jpg"
                    frame_path = os.path.join(gif_output_folder, frame_filename)
                    img.save(frame_path, quality=85)
                    frame_url = f"/static/gifs/{video_id}/{frame_filename}"
                    frame_urls.append(frame_url)
                    if idx == 0:
                        thumbnail_url = frame_url

                return f"/static/gifs/{video_id}/{gif_filename}", frame_urls, thumbnail_url
            return None, [], None

        for segment in accepted_segments:
            gif_url, frame_urls, thumbnail_url = save_frames_and_gif(segment, 'accepted')
            segment_data.append({
                "id": segment_id,
                "type": "accepted",
                "gif": gif_url,
                "frames": frame_urls,
                "thumbnail": thumbnail_url,
                "start": segment["start"],
                "end": segment["end"],
                "reasons": []
            })
            segment_id += 1

        for segment in rejected_segments:
            gif_url, frame_urls, thumbnail_url = save_frames_and_gif(segment, 'rejected')
            segment_data.append({
                "id": segment_id,
                "type": "rejected",
                "gif": gif_url,
                "frames": frame_urls,
                "thumbnail": thumbnail_url,
                "start": segment["start"],
                "end": segment["end"],
                "reasons": list(segment["reasons"])
            })
            segment_id += 1

        feedback_data = {
            "video_id": video_id,
            "score": final_score,
            "feedbacks": list(feedbacks),
            "segments": segment_data,
            "processed_video_url": url_for('uploaded_file', filename=f"{video_id}_processed_{filename}")
        }

        with open(os.path.join(gif_output_folder, 'feedback.json'), 'w') as f:
            json.dump(feedback_data, f)

        # Save to database
        new_analysis = Analysis(
            video_id=video_id,
            score=final_score,
            feedback="; ".join(feedbacks)
        )
        db.session.add(new_analysis)
        db.session.commit()

        return redirect(url_for('feedback_page', video_id=video_id))

    return render_template('upload.html')



@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()
    return render_template("history.html", analyses=analyses)



@app.route('/api/feedback/<video_id>')
def api_feedback(video_id):
    feedback_file = os.path.join(app.config['GIF_FOLDER'], video_id, 'feedback.json')
    if not os.path.exists(feedback_file):
        return jsonify({"error": "Feedback not found"}), 404

    with open(feedback_file, 'r') as f:
        data = json.load(f)

    accepted_segments = []
    rejected_segments = []

    for seg in data['segments']:
        formatted = {
            "gif_url": seg['gif'],
            "frames": seg.get('frames', []),
            "start": seg['start'],
            "end": seg['end'],
            "thumbnail": seg.get("thumbnail")
        }
        if seg["type"] == "accepted":
            accepted_segments.append(formatted)
        else:
            formatted["reasons"] = seg["reasons"]
            rejected_segments.append(formatted)

    return jsonify({
        "final_score": data['score'],
        "feedbacks": data['feedbacks'],
        "accepted_segments": accepted_segments,
        "rejected_segments": rejected_segments
    })


@app.route('/feedback/<video_id>')
def feedback_page(video_id):
    feedback_file = os.path.join(app.config['GIF_FOLDER'], video_id, 'feedback.json')
    if not os.path.exists(feedback_file):
        return 'Feedback not found', 404

    with open(feedback_file, 'r') as f:
        data = json.load(f)

    return render_template('feedback.html',
                           video_id=video_id,
                           score=data['score'],
                           feedbacks=data['feedbacks'],
                           segments=data['segments'],
                           processed_video_url=data.get('processed_video_url'))


@app.route('/export_pdf/<video_id>')
def export_pdf(video_id):
    feedback_file = os.path.join(app.config['GIF_FOLDER'], video_id, 'feedback.json')
    if not os.path.exists(feedback_file):
        return 'Feedback not found', 404

    with open(feedback_file, 'r') as f:
        data = json.load(f)

    pdf_filename = f"{video_id}_report.pdf"
    pdf_path = os.path.join(app.config['GIF_FOLDER'], video_id, pdf_filename)

    generate_pdf_report(
        feedbacks=data['feedbacks'],
        score=data['score'],
        accepted_segments=[
            {"start": s["start"], "end": s["end"], "thumbnail": s.get("thumbnail")}
            for s in data['segments'] if s["type"] == "accepted"
        ],
        rejected_segments=[
            {"start": s["start"], "end": s["end"], "reasons": s["reasons"], "thumbnail": s.get("thumbnail")}
            for s in data['segments'] if s["type"] == "rejected"
        ],
        pdf_path=pdf_path
    )

    return send_from_directory(os.path.join(app.config['GIF_FOLDER'], video_id), pdf_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
