
from flask import Flask, render_template, request, redirect, url_for, send_file
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
import face_recognition
import time
import pandas as pd
from datetime import datetime, timezone
import csv
import mindwave
import os
from flask_socketio import SocketIO
import threading



# Flask App Initialization
name = "Attention"
app = Flask(__name__)
socketio = SocketIO(app)

filename = None

app.config['GRAPH_DIR'] = './static/graphs'
os.makedirs(app.config['GRAPH_DIR'], exist_ok=True)

@app.route("/Analytics", methods=['GET','POST'])
def Analytics():
    # Load the CSV file
    csv_path = './Custom/Evaluation.csv'  # Replace with your CSV file path
    data = pd.read_csv(csv_path)

    # Process the data (assume columns like '30/01/2025', '31/01/2025', etc.)
    data['t_focused'] = data['t_focused'].fillna(0).astype(int)
    data['t_distracted'] = data['t_distracted'].fillna(0).astype(int)

    # Calculate total_time
    data['t_total'] = data['t_focused'] + data['t_distracted']

    # Pivot the data if necessary to convert date columns to rows (for plotting)
    # Assuming columns are named as dates (e.g., '30/01/2025')
    date_columns = [col for col in data.columns if '/' in str(col)]  # Columns with dates

    # Convert the data to JSON for the frontend (to keep the rows as students)
    students = data.to_dict(orient='records')

    today_date = datetime.today().strftime('%d/%m/%Y') # Ensure this matches the column name in CSV

    if today_date not in data.columns:
        return f" Attendance for today's date '{today_date}' not calculated"
    
    present_count = (data[today_date] == 'Present').sum()
    absent_count = (data[today_date] == 'Absent').sum()

    # Generate graphs
    generate_focus_vs_distraction_chart(data, date_columns)
    generate_attendance_chart(present_count, absent_count)
    generate_class_engagement_chart(data)

    # Pass the data and image paths to the template
    return render_template('analytics.html', students=students, 
                           focus_chart_url='/static/graphs/focus_vs_distraction.png',
                           attendance_chart_url='/static/graphs/attendance_chart.png',
                           class_engagement_chart_url='/static/graphs/class_engagement_chart.png')


def generate_focus_vs_distraction_chart(data, date_columns):
    # Calculate total focused and distracted time
    total_focused_time = data['t_focused'].sum()
    total_distracted_time = data['t_distracted'].sum()

    # Create a doughnut chart
    labels = ['Focused Time', 'Distracted Time']
    sizes = [total_focused_time, total_distracted_time]
    colors = ['#4CAF50', '#FF5733']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.4})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title("Focus vs Distraction")
    
    # Save the chart
    fig.savefig(os.path.join(app.config['GRAPH_DIR'], 'focus_vs_distraction.png'))
    plt.close(fig)



def generate_attendance_chart(present_count, absent_count):
    """Generate a pie chart for student attendance with updated colors and count display."""

    labels = [
        f'Present: {present_count} ({present_count / (present_count + absent_count) * 100:.1f}%)',
        f'Absent: {absent_count} ({absent_count / (present_count + absent_count) * 100:.1f}%)'
    ]
    
    sizes = [present_count, absent_count]
    colors = ['#3498db', '#e67e22']  # Blue & Orange for better visibility

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='', startangle=90, wedgeprops={'width': 0.4}
    )
    
    # Manually add text labels to show counts inside the pie chart
    for i, autotext in enumerate(autotexts):
        autotext.set_text(f"{sizes[i]}")  # Set count inside the chart
        autotext.set_color('white')       # Make text readable
        autotext.set_fontsize(12)

    ax.axis('equal')  # Ensure the pie chart is a perfect circle
    ax.set_title("Attendance Distribution", fontsize=14, fontweight='bold')

    # Ensure directory exists before saving
    os.makedirs(app.config['GRAPH_DIR'], exist_ok=True)

    # Save the chart
    chart_path = os.path.join(app.config['GRAPH_DIR'], 'attendance_chart.png')
    fig.savefig(chart_path, bbox_inches='tight')

    plt.close(fig)



def generate_class_engagement_chart(data):
    # Prepare data for the bar chart
    names = data['Name']
    focused_times_by_student = data['t_focused']

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, focused_times_by_student, color='#4CAF50')
    ax.set_title("Class Engagement by Student")
    ax.set_xlabel("Student Name")
    ax.set_ylabel("Total Focused Time (minutes)")
    
    # Save the chart
    fig.savefig(os.path.join(app.config['GRAPH_DIR'], 'class_engagement_chart.png'))
    plt.close(fig)



@app.route("/Messages", methods=['GET','POST'])
def Messages():
    return render_template('messages.html')

@app.route("/Settings", methods=['GET','POST'])
def Settings():
    return render_template('setting.html')

@app.route("/", methods=['GET','POST'])
def home():
    return render_template('main.html')
# Routes
@app.route('/live', methods=['GET', 'POST'])
def live():
    return render_template('live.html')

@app.route('/eeg', methods=['GET', 'POST'])
def eeg():
    return render_template('eeg.html')

@app.route("/start_recording", methods=["POST"])
def start_recording():
    name = request.form["name1"]  # Get roll number from the form
    print(f"Received Name: {name}")

    threading.Thread(target=record_eeg_data, args=(name,)).start()

    return redirect(url_for("raw"))
@app.route("/raw", methods=['GET','POST'])
def raw():
    return render_template("raw.html", message="Recording is in progress!")



stop_recording = False  # Global flag to stop recording

def record_eeg_data(name):
    global stop_recording
    stop_recording = False  

    ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')  # Generate filename
    filename = f'{name}_{ts}.csv'

    print(f"Recording session data to {filename} for Name: {name}")

    # Connect to Mindwave
    print("Connecting to Mindwave...")
    try:
        headset = mindwave.Headset('COM4')
        time.sleep(10)  # Wait for data stream
    except Exception as e:
        print(f"Error connecting to headset: {e}")
        return

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Raw', 'Attention', 'Meditation', 'delta', 'theta', 'low-alpha',
                             'high-alpha', 'low-beta', 'high-beta', 'low-gamma', 'mid-gamma', 'Attention Level'])

            low_attention_count = 0
            medium_attention_count = 0
            high_attention_count = 0

            iteration_time = 0.5  # Interval time

            
            while not stop_recording:  
                ts = datetime.now(timezone.utc).isoformat()
                try:
                    attention = headset.attention
                    raw_value = headset.raw_value
                    meditation = headset.meditation
                    waves = headset.waves if headset.waves else {}

                    # Attention Level Calculation
                    if attention <= 30:
                        attention_level = 0
                        low_attention_count += 1
                    elif 31 <= attention <= 70:
                        attention_level = 1
                        medium_attention_count += 1
                    else:
                        attention_level = 2
                        high_attention_count += 1

                    eeg_data = {
                        "timestamp": ts,
                        "raw_value": raw_value,
                        "attention": attention,
                        "attention_level": attention_level,
                        "meditation": meditation,
                        "waves": waves
                    }
                    socketio.emit('eeg_data', eeg_data)

                    values = [ts, raw_value, attention, meditation] + list(waves.values()) + [attention_level]
                    writer.writerow(values)

                    time.sleep(iteration_time)

                except Exception as e:
                    print(f"Error reading EEG data: {e}")

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        # Calculate attention time
        total_time = (low_attention_count + medium_attention_count + high_attention_count) * iteration_time
        low_attention_time = low_attention_count * iteration_time
        medium_attention_time = medium_attention_count * iteration_time
        high_attention_time = high_attention_count * iteration_time

        # Send summary to webpage
        summary = {
            "low_attention_time": low_attention_time,
            "medium_attention_time": medium_attention_time,
            "high_attention_time": high_attention_time,
            "total_time": total_time,
        }
        socketio.emit('session_summary', summary)

        # Append to class summary file
        class_filename = r'Custom\class_attention_times.csv'
        file_exists = os.path.isfile(class_filename)

        with open(class_filename, 'a', newline='') as class_file:
            class_writer = csv.writer(class_file)
            if not file_exists:
                class_writer.writerow(['Name', 'High Attention Time', 'Medium Attention Time', 'Low Attention Time', 'Total Time'])
            class_writer.writerow([name, high_attention_time, medium_attention_time, low_attention_time, total_time])

        print(f"\nAttention times for Name: {name} added to {class_filename}.")

        # ✅ Correct way to close the headset connection
        try:
            if hasattr(headset, "stop"):
                headset.stop()
            elif hasattr(headset, "serial"):
                headset.serial.close()
            print("Headset connection closed successfully.")
        except Exception as e:
            print(f"Error closing headset: {e}")

    

# Function to stop recording dynamically
@socketio.on('stop_recording')
def stop_eeg_recording():
    global stop_recording
    stop_recording = True
    print("Received stop signal from client.")


@app.route('/download-report')
def download_report():
    try:
        # Define file paths
        image_report_path = os.path.join(os.getcwd(), 'Custom', 'Evaluation.csv')
        eeg_report_path = os.path.join(os.getcwd(), 'Custom', 'class_attention_times.csv')
        final_report_path = os.path.join(os.getcwd(), 'Custom', 'final_report.csv')

        # Load the reports with error handling
        image_report = pd.read_csv(image_report_path, delimiter=',', on_bad_lines='skip')
        eeg_report = pd.read_csv(eeg_report_path, delimiter=',', on_bad_lines='skip')

        # # Print headers for debugging
        # print(image_report.head())
        # print(eeg_report.head())

        image_report['Name'] = image_report['Name'].str.strip().astype(str)
        eeg_report['Name'] = eeg_report['Name'].str.strip().astype(str)


        # Merge on 'Name'
        combined_report = pd.merge(image_report, eeg_report, on='Name', how='inner')

        # Avoid division by zero
        combined_report['attention_image'] = (combined_report['t_focused'] / combined_report['t_total']) * 100
        combined_report['attention_image'].fillna(0, inplace=True)
        denominator = combined_report['High Attention Time'] + combined_report['Medium Attention Time'] + combined_report['Low Attention Time']
        combined_report['attention_EEG'] = (combined_report['High Attention Time'] * 1 + combined_report['Medium Attention Time'] * 0.5) / denominator * 100
        combined_report['attention_EEG'].fillna(0, inplace=True)  # Prevent NaN

        # Weighted final attention score
        combined_report['Final Attention (%)'] = 0.6 * combined_report['attention_EEG'] + 0.4 * combined_report['attention_image']

        # Save the final report
        combined_report.to_csv(final_report_path, index=False)

        return send_file(final_report_path, as_attachment=True, download_name='Report.csv', mimetype='text/csv')
    
    except Exception as e:
        print(f"Error while downloading the file: {e}")
        return "Failed to download the report. Please try again later", 500

    
@app.route('/done', methods=['GET', 'POST'])
def done():
    if request.method == "POST":
        name1 = request.form["name1"]

        # Current Date and Time
        now = datetime.now()
        date = str(now.strftime("%d-%m-%Y %H:%M")).split(' ')[0].replace('-', '/').encode()

        # Load known face encodings and names
        face_data = [
            ("prasad", "images/prasad.jpg"),
            ("atharva", "images/atharva.jpg"),
            ("vaibhav", "images/vaibhav.jpg"),
            ("mrinmayee", "images/mrinmayee.jpg"),
            ("deepali", "images/deepali.jpg"),
            ("chaitanya", "images/chaitanya.jpg"),
        ]

        known_face_encodings = []
        known_face_names = []
        for name, path in face_data:
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)

        # Initialize tracking and attendance data
        t_students = {name: {'focus': 0, 'distract': 0, 'attendance': 0} for name in known_face_names}
        df = pd.read_csv('Custom/Evaluation.csv')

        # Add today's date column if not exists
        today_date = datetime.now().strftime('%d/%m/%Y')
        if today_date not in df.columns:
            df[today_date] = 'Absent'

        # Variables
        face_locations = []
        face_encodings = []
        process_this_frame = True
        attendance = []
        img_rows, img_cols = 48, 48
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        box_color = (255, 245, 152)

        # Load Models
        model = []
        print('Loading Models...')
        for i in range(2):
            m = load_model(f'saved_model/cnn{i}.h5')
            model.append(m)
            print(f'Model {i + 1}/3 loaded')
        m = load_model('saved_model/ensemble.h5')
        model.append(m)
        print('Ensemble model loaded\nLoading complete!')

        # Prediction Function
        def predict(x):
            x_rev = np.flip(x, 1)
            x = x.astype('float32') / 255
            x_rev = x_rev.astype('float32') / 255
            p = np.zeros((1, 14))
            p[:, :7] = model[0].predict(x.reshape(1, 48, 48, 1))
            p[:, 7:] = model[1].predict(x_rev.reshape(1, 48, 48, 1))
            return model[2].predict(p)

        # Tracking states
        t_states = {name: {'focus_start': None, 'distract_start': None} for name in known_face_names}

        # Video Capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.open()

        start_session_time = time.time()

        while True:
            ret, img = cap.read()
            curTime = time.time()

            # Get Faces
            faces = get_faces(img, method='haar')
            for i, (face, x, y, w, h) in enumerate(faces):
                pre = predict(face)
                emotion_index = np.argmax(pre)
                emotion_label = emotion_labels[emotion_index]
                emotion_confidence = int(pre[0, emotion_index] * 100)

                name = ''
                try:
                    small_frame = cv2.resize(img[y-20:y+h+20, x-20:x+w+20], (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    if process_this_frame:
                     face_locations = face_recognition.face_locations(small_frame)
                     if face_locations:
                         face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                         for face_encoding in face_encodings:
                             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                             name = "Unknown"

                             if True in matches:
                                 # Use face_distance to find the best match
                                 face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                 best_match_index = np.argmin(face_distances)

                                 # Ensure the match is correct
                                 if matches[best_match_index]:
                                     name = known_face_names[best_match_index]
                                     confidence_score = 1 - face_distances[best_match_index]  # Higher means better match
                                     print(f"Matched {name} with confidence {confidence_score:.2f}")

                                     t_students[name]['attendance'] = 1
                                     if name not in attendance:
                                        attendance.append(name)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

                # Process focus and distraction times
                if name != "Unknown" and name:
                    if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']:
                        if t_states[name]['focus_start'] is not None:
                            focus_duration = curTime - t_states[name]['focus_start']
                            t_students[name]['focus'] += focus_duration
                            t_states[name]['focus_start'] = None
                        if t_states[name]['distract_start'] is None:
                            t_states[name]['distract_start'] = curTime
                    else:
                        if t_states[name]['distract_start'] is not None:
                            distract_duration = curTime - t_states[name]['distract_start']
                            t_students[name]['distract'] += distract_duration
                            t_states[name]['distract_start'] = None
                        if t_states[name]['focus_start'] is None:
                            t_states[name]['focus_start'] = curTime

                # Draw Bounding Boxes and Labels
                tl = (x, y)
                br = (x + w, y + h)
                coords = (x, y - 2)

                box_color = (0, 0, 255) if emotion_label in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'] else (255, 245, 152)
                txt = f"{name} {emotion_label} [{emotion_confidence}%] | {'Distracted' if emotion_label != 'Neutral' else 'Focused'}"
                img = cv2.rectangle(img, tl, br, box_color, 2)
                cv2.putText(img, txt, coords, font, 0.8, text_color, 1, cv2.LINE_AA)

            # Display
            cv2.imshow('Camera', img)

            # Quit on 'q'
            if cv2.waitKey(20) & 0xFF == ord('q'):
                end_session_time = time.time()
                total_session_time = end_session_time - start_session_time

                for name in attendance:
                    if name in t_students:
                        focus_time = t_students[name]['focus']
                        distract_time = t_students[name]['distract']
                        if pd.isna(df.loc[df['Name'] == name, 't_focused']).any():
                            df.loc[df['Name'] == name, 't_focused'] = 0.0
                        if pd.isna(df.loc[df['Name'] == name, 't_distracted']).any():
                            df.loc[df['Name'] == name, 't_distracted'] = 0.0

                        df.loc[df['Name'] == name, 't_focused'] += focus_time
                        df.loc[df['Name'] == name, 't_distracted'] += distract_time
                        df.loc[df['Name'] == name, 't_total'] = df.loc[df['Name'] == name, 't_focused'] + df.loc[df['Name'] == name, 't_distracted']
                        df.loc[df['Name'] == name, today_date] = 'Present'

                df.loc[~df['Name'].isin(attendance), today_date] = 'Absent'
                df.to_csv('Custom/Evaluation.csv', index=False)
                break

        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('live'))

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000)