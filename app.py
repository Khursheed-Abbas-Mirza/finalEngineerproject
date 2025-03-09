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
            "Peanut-butter.jpg",
            "milk.jpg",
            "almonds.jpeg",
            "avacado.jpg",
            "brown-rice.jpg",
            "Bananas.jpeg",
            "cheese.jpg",
            "Granola.jpeg",
            "Chicken-thighs.jpg",
            "dried.jpg"
        ],
        "food_type": "Eat more healthy fats, proteins, and energy-rich foods like nuts, dairy, and lean meats to gain weight.",
        "exercises": [
            "Weightlifting.gif",
            "Push-ups.gif",
            "Squats.gif",
            "Pull-ups.gif",
            "BenchPress.gif",
            "Deadlifts.jpeg",
            "Lunges.gif",
        ],
        "exercise_type": "Focus on strength training and weight lifting to build muscle and gain healthy weight.",
        "Goal": "Build muscle mass"
    },
    "Normal": {
        "foods": [
            "Grilledchicken.jpg",
            "Quinoa.jpeg",
            "sweet-potato.jpg",
            "Spinach.jpg",
            "Salmon.jpeg",
            "Greek-yogurt.jpg",
            "WholeGrainBread.jpg",
            "Broccoli.jpeg",
            "Oliveoil.jpg",
            "Eggs.jpeg"
        ],
        "food_type": "Eat a mix of proteins, vegetables, and whole grains to stay healthy and fit.",
        "exercises": [
            "BriskWalking.jpeg",
            "Jogging.jpg",
            "Cycling.jpeg",
            "Squats.gif",
            "Plank.jpeg",
            "Yoga.jpeg",
            "jumping-jacks.gif",
        ],
        "exercise_type": "Maintain a mix of cardio, strength, and flexibility exercises for overall health.",
        "Goal": "Maintain fitness"
    },
    "Overweight": {
        "foods": [
            "leefy.jpg",
            "Leanturkey.jpeg",
            "Cucumber.jpg",
            "Berries.jpg",
            "Lentils.jpg",
            "Chiaseeds.jpg",
            "Zucchini.jpeg",
            "Cauliflower.jpg",
            "Chicken-Breast.jpg",
            "Almondmilk.jpeg"
        ],
        "food_type": "Eat more vegetables, lean meats, and fiber-rich foods while avoiding sugary and fried foods.",
        "exercises": [
            "Jogging.jpg",
            "Cycling.jpeg",
            "swimming.jpeg",
            "stairclimbing.jpg",
            "Burpees.gif",
            "walkinglunges.gif",
            "jumprope.gif",
        ],
        "exercise_type": "Increase cardio workouts like jogging, cycling, and swimming to burn excess fat.",
        "Goal": "Burn calories"
    },
    "Obese": {
        "foods": [
            "Broccoli.jpeg",
            "Eggwhites.jpg",
            "Oats.jpeg",
            "Cauliflower.jpg",
            "Tuna-Fish.jpg",
            "GreenBeans.jpeg",
            "Apples.jpg",
            "Kale.jpg",
            "turkey-meatballs.jpg",
            "Cottagecheese.jpeg"
        ],
        "food_type": "Focus on high-fiber foods like vegetables, fruits, and whole grains while reducing unhealthy fats and sugars.",
        "exercises": [
            "walking.jpg",
            "wateraerobics.jpeg",
            "seated-leg.gif",
            "chairsquats.gif",
            "armcircle.gif",
            "stretch.gif",
            "stepup.gif",
        ],
        "exercise_type": "Start with light exercises like walking, stretching, and water aerobics to avoid joint strain.",
        "Goal": "Sustainable weight loss & health"
    }
}

    
    return {
        "success": True,
        "category": standardized_category,
        "current_bmi": round(bmi, 1),
        "ideal_weight_range": f"{round(min_ideal_weight, 1)}–{round(max_ideal_weight, 1)} kg",
        "food_type": recommendations[standardized_category]["food_type"],
        "food_recommendations": recommendations[standardized_category]["foods"],
        "exercise_type": recommendations[standardized_category]["exercise_type"],
        "exercise_recommendations": recommendations[standardized_category]["exercises"],
        "goal": recommendations[standardized_category]["Goal"]
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

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)