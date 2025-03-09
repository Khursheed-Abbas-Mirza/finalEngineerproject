from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np
import warnings
from flask_cors import CORS
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
model = joblib.load("obesity.pkl")  # Load the new model

def get_recommendations(age, height, weight, gender):
    # Calculate BMI if not provided
 
    height_m = height / 100  # Convert cm to meters
    bmi = weight / (height_m * height_m)
    
    # Predict category
    input_data = np.array([[age, height, weight, gender, bmi]])
    category = model.predict(input_data)
    category = category[0]
    # Calculate ideal weight range
    height_m = height/100
    min_ideal_weight = 18.5 * (height_m * height_m)
    max_ideal_weight = 24.9 * (height_m * height_m)
    
    # Adjust category names to match dataset
    category_map = {
        "Underweight": "Underweight",
        "Normal weight": "Normal",
        "Overweight": "Overweight",
        "Obese": "Obese"
    }
    standardized_category = category_map[category]
    
    # Simplified food names (10 per category) and exercises (7 per category)
    recommendations = {
        "Underweight": {
            "foods": [
                "Peanut butter",
                "Whole milk",
                "Almonds",
                "Avocado",
                "Brown rice",
                "Bananas",
                "Cheese",
                "Granola",
                "Chicken thighs",
                "Dried fruit"
            ],
            "exercises": [
                "Weightlifting.gif",
                "Push-ups.gif",
                "Squats.gif",
                "Pull-ups.gif",
                "BenchPress.gif",
                "Deadlifts.jpeg",
                "Lunges.gif",
            ],
            "Goal": "Build muscle mass"
        },
        "Normal": {
            "foods": [
                "Grilled chicken",
                "Quinoa",
                "Sweet potatoes",
                "Spinach",
                "Salmon",
                "Greek yogurt",
                "Whole-grain bread",
                "Broccoli",
                "Olive oil",
                "Eggs"
            ],
            "exercises": [
                "BriskWalking.jpeg",
                "Jogging.jpg",
                "Cycling.jpeg",
                "Squats.gif",
                "Plank.jpeg",
                "Yoga.jpeg",
                "jumping-jacks.gif",
            ],
        "Goal":" Maintain fitness"
        },
        "Overweight": {
            "foods": [
                "Leafy greens",
                "Lean turkey",
                "Cucumber",
                "Berries",
                "Lentils",
                "Chia seeds",
                "Zucchini",
                "Cauliflower rice",
                "Chicken breast",
                "Almond milk"
            ],
            "exercises": [
                "Jogging.jpg",
                "Cycling.jpeg",
                "swimming.jpeg",
                "stairclimbing.jpg",
                "Burpees.gif",
                "walkinglunges.gif",
                "jumprope.gif",
            ]
                ,"Goal": "Burn calories"
        },
        "Obese": {
            "foods": [
                "Broccoli",
                "Egg whites",
                "Oats",
                "Cauliflower",
                "Tuna",
                "Green beans",
                "Apples",
                "Kale",
                "Turkey meatballs",
                "Cottage cheese"
            ],
            "exercises": [
                "walking.jpg",
                "wateraerobics",
                "seated-leg.gif",
                "chairsquats.gif",
                "armcircle.gif",
                "stretch.gif",
                "stepup.gif",
            ],
                "Goal": "Start slow"
        }
    }
    
    return {
        "success": True,
        "category": standardized_category,
        "current_bmi": round(bmi, 1),
        "ideal_weight_range": f"{round(min_ideal_weight, 1)}–{round(max_ideal_weight, 1)} kg",
        "food_recommendations": recommendations[standardized_category]["foods"],
        "exercise_recommendations": recommendations[standardized_category]["exercises"]
    }

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Extract inputs, with BMI optional
    age = data["age"]
    height = data["height"]
    weight = data["weight"]
    gender =0  if data["gender"]=='Male' else 1 # 0 for Male, 1 for Female
    
    result = get_recommendations(age, int(height), int(weight), gender)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True,host="localhost",port=5000)