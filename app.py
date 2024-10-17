import io
from flask import Flask, jsonify, render_template, request, redirect, send_file, send_from_directory, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
from models import User
from flask_migrate import Migrate
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.config['SECRET_KEY'] = '878965833497b302f30872f5fefcd6e0'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
migrate = Migrate(app, db)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('overview_page'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing_page'))

# Load the sentiment analysis model
model = joblib.load('C:\\Users\\Khadijah\\fyp\\model.pkl')

@app.route('/')
def landing_page():
    return render_template('LandingPage.html')

@app.route('/OverviewPage.html')
def overview_page():
    return render_template('OverviewPage.html')

@app.route('/jntPage.html')
def jnt_page():
    return render_template('jntPage.html')

@app.route('/spxPage.html')
def spx_page():
    return render_template('spxPage.html')

@app.route('/dhlPage.html')
def dhl_page():
    return render_template('dhlPage.html')

@app.route('/ComparePage.html')
def compare_page():
    return render_template('ComparePage.html')

@app.route('/ReportPage.html')
def report_page():
    return render_template('ReportPage.html')

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    selected_courier = request.form['courier']
    
    # Map courier names to their respective PDF filenames
    courier_to_pdf = {
        'J&T Express': 'jnt.pdf',
        'SPX Xpress': 'spx.pdf',
        'DHL Express': 'dhl.pdf',
    }
    
    if selected_courier in courier_to_pdf:
        # Get the corresponding PDF filename
        pdf_filename = courier_to_pdf[selected_courier]
        pdf_path = f'/Users/Khadijah/fyp/templates/{pdf_filename}'  # Replace with the actual path
        
        # Check if user is authorized to generate reports for selected courier
        if selected_courier != current_user.name:
            flash('You are not authorized to generate reports for other couriers.', 'error')
            return redirect(url_for('report_page'))
        
        # Flash success message
        flash(f'Report generated successfully!', 'success')
        
        # Send the pre-generated PDF file to the user for download
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    else:
        flash('Invalid request.', 'error')
        return redirect(url_for('report_page'))

# Load model
with open('model.pkl', 'rb') as file:
    model = joblib.load(file)

# Load the TF-IDF vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Global variable to store the in-memory file
download_file = None

@app.route('/AnalyzerPage.html', methods=['GET', 'POST'])
def analyzer_page():
    global download_file
    success_message = None
    df_html = None
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    pie_chart_html = None  
    bar_chart_html = None 

    if request.method == 'POST':
        if 'sentence' in request.form:
            sentence = request.form['sentence']
            input_data_tfidf = vectorizer.transform([sentence])
            sentiment_index = model.predict(input_data_tfidf)[0]
            sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment_result = sentiment_labels[sentiment_index]
            sentiment_counts[sentiment_result] += 1
            return render_template('AnalyzerPage.html', sentence=sentence, sentiment_result=sentiment_result, sentiment_counts=sentiment_counts)

        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
                return redirect(request.url)
            
            if file:
                file_path = os.path.join(file.filename)
                file.save(file_path)
                
                df = pd.read_csv(file_path)
                df['sentiment'] = df['text'].apply(lambda x: model.predict(vectorizer.transform([x]))[0])
                sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                df['sentiment'] = df['sentiment'].map(sentiment_labels)
                
                # Count the sentiments
                sentiment_counts = df['sentiment'].value_counts().to_dict()

                # Save the dataframe to a CSV file
                df.to_csv('results.csv', index=False)

                # Convert sentiment counts to JSON format
                sentiment_counts_json = jsonify(sentiment_counts)

                df_html = df.to_html(classes='table table-striped', index=False)

                # Save the dataframe to a BytesIO object
                output = io.BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)
                
                # Save the in-memory file to a global variable
                download_file = output

                os.remove(file_path)  # Clean up the saved file after processing

                 # Create a pie chart
                pie_fig = px.pie(values=list(sentiment_counts.values()), 
                                 names=list(sentiment_counts.keys()), 
                                 title='Sentiment Distribution')
                pie_fig.update_traces(marker=dict(colors=['#4CAF50', '#F44336', '#B0B0B0']))  # Red, Grey, Green
                pie_fig.update_layout(title={'text': 'Sentiment Distribution', 'x': 0.5, 'xanchor': 'center'})
                pie_chart_html = pie_fig.to_html(full_html=False)

                # Create a bar chart
                bar_fig = go.Figure(data=[go.Bar(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), 
                                                 marker_color=['#4CAF50', '#F44336', '#B0B0B0'])])  # Green, Grey, Red
                bar_fig.update_layout(title={'text': 'Sentiment Count', 'x': 0.5, 'xanchor': 'center'}, 
                                      xaxis_title='Sentiment', 
                                      yaxis_title='Count')
                bar_chart_html = bar_fig.to_html(full_html=False)

    return render_template('AnalyzerPage.html', success_message=success_message, df_html=df_html, download_link=False, pie_chart_html=pie_chart_html, bar_chart_html=bar_chart_html)


@app.route('/download', methods=['GET'])
def download():
    global download_file
    if download_file:
        return send_file(download_file, as_attachment=True, download_name='results.csv', mimetype='text/csv')
    return redirect('/AnalyzerPage.html')


@app.route('/overview1.html')
def overview1():
    return send_from_directory('templates', 'overview1.html')

@app.route('/overview2.html')
def overview2():
    return send_from_directory('templates', 'overview2.html')

@app.route('/overview3.html')
def overview3():
    return send_from_directory('templates', 'overview3.html')

@app.route('/jnt1.html')
def jnt1():
    return send_from_directory('templates', 'jnt1.html')

@app.route('/jnt2.html')
def jnt2():
    return send_from_directory('templates', 'jnt2.html')

@app.route('/jnt3.html')
def jnt3():
    return send_from_directory('templates', 'jnt3.html')

@app.route('/jnt4.html')
def jnt4():
    return send_from_directory('templates', 'jnt4.html')

@app.route('/jnt5.html')
def jnt5():
    return send_from_directory('templates', 'jnt5.html')

@app.route('/jnt6.html')
def jnt6():
    return send_from_directory('templates', 'jnt6.html')

@app.route('/jnt7.html')
def jnt7():
    return send_from_directory('templates', 'jnt7.html')

@app.route('/jnt8.html')
def jnt8():
    return send_from_directory('templates', 'jnt8.html')

@app.route('/jnt9.html')
def jnt9():
    return send_from_directory('templates', 'jnt9.html')

@app.route('/jnt10.html')
def jnt10():
    return send_from_directory('templates', 'jnt10.html')

@app.route('/jnt11.html')
def jnt11():
    return send_from_directory('templates', 'jnt11.html')

@app.route('/jnt12.html')
def jnt12():
    return send_from_directory('templates', 'jnt12.html')

@app.route('/jnt13.html')
def jnt13():
    return send_from_directory('templates', 'jnt13.html')

@app.route('/dhl1.html')
def dhl1():
    return send_from_directory('templates', 'dhl1.html')

@app.route('/dhl2.html')
def dhl2():
    return send_from_directory('templates', 'dhl2.html')

@app.route('/dhl3.html')
def dhl3():
    return send_from_directory('templates', 'dhl3.html')

@app.route('/dhl4.html')
def dhl4():
    return send_from_directory('templates', 'dhl4.html')

@app.route('/dhl5.html')
def dhl5():
    return send_from_directory('templates', 'dhl5.html')

@app.route('/dhl6.html')
def dhl6():
    return send_from_directory('templates', 'dhl6.html')

@app.route('/dhl7.html')
def dhl7():
    return send_from_directory('templates', 'dhl7.html')

@app.route('/dhl8.html')
def dhl8():
    return send_from_directory('templates', 'dhl8.html')

@app.route('/dhl9.html')
def dhl9():
    return send_from_directory('templates', 'dhl9.html')

@app.route('/dhl10.html')
def dhl10():
    return send_from_directory('templates', 'dhl10.html')

@app.route('/dhl11.html')
def dhl11():
    return send_from_directory('templates', 'dhl11.html')

@app.route('/dhl12.html')
def dhl12():
    return send_from_directory('templates', 'dhl12.html')

@app.route('/dhl13.html')
def dhl13():
    return send_from_directory('templates', 'dhl13.html')

@app.route('/spx1.html')
def spx1():
    return send_from_directory('templates', 'spx1.html')

@app.route('/spx2.html')
def spx2():
    return send_from_directory('templates', 'spx2.html')

@app.route('/spx3.html')
def spx3():
    return send_from_directory('templates', 'spx3.html')

@app.route('/spx4.html')
def spx4():
    return send_from_directory('templates', 'spx4.html')

@app.route('/spx5.html')
def spx5():
    return send_from_directory('templates', 'spx5.html')

@app.route('/spx6.html')
def spx6():
    return send_from_directory('templates', 'spx6.html')

@app.route('/spx7.html')
def spx7():
    return send_from_directory('templates', 'spx7.html')

@app.route('/spx8.html')
def spx8():
    return send_from_directory('templates', 'spx8.html')

@app.route('/spx9.html')
def spx9():
    return send_from_directory('templates', 'spx9.html')

@app.route('/spx10.html')
def spx10():
    return send_from_directory('templates', 'spx10.html')

@app.route('/spx11.html')
def spx11():
    return send_from_directory('templates', 'spx11.html')

@app.route('/spx12.html')
def spx12():
    return send_from_directory('templates', 'spx12.html')

@app.route('/spx13.html')
def spx13():
    return send_from_directory('templates', 'spx13.html')

@app.route('/compare1.html')
def compare1():
    return send_from_directory('templates', 'compare1.html')

@app.route('/compare2.html')
def compare2():
    return send_from_directory('templates', 'compare2.html')

@app.route('/compare3.html')
def compare3():
    return send_from_directory('templates', 'compare3.html')

@app.route('/compare4.html')
def compare4():
    return send_from_directory('templates', 'compare4.html')

@app.route('/compare5.html')
def compare5():
    return send_from_directory('templates', 'compare5.html')

@app.route('/compare6.html')
def compare6():
    return send_from_directory('templates', 'compare6.html')

@app.route('/compare7.html')
def compare7():
    return send_from_directory('templates', 'compare7.html')

@app.route('/compare8.html')
def compare8():
    return send_from_directory('templates', 'compare8.html')

@app.route('/compare9.html')
def compare9():
    return send_from_directory('templates', 'compare9.html')

@app.route('/compare10.html')
def compare10():
    return send_from_directory('templates', 'compare10.html')

@app.route('/compare11.html')
def compare11():
    return send_from_directory('templates', 'compare11.html')

@app.route('/compare12.html')
def compare12():
    return send_from_directory('templates', 'compare12.html')

@app.route('/compare13.html')
def compare13():
    return send_from_directory('templates', 'compare13.html')



if __name__ == '__main__':
    app.run(debug=True, port=8000)
