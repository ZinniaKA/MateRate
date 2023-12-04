from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quizapp.db'
db = SQLAlchemy(app)

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(50), nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    answer_text = request.form.get('answer')
    new_answer = Answer(answer=answer_text)
    db.session.add(new_answer)
    db.session.commit()
    return jsonify({'result': f'Your favourite game is: {answer_text}'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
