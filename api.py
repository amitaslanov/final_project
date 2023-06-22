from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    print (data)
    room_number = int(data['room_number'])
    area = float(data['Area'])
    hasElevator = int(data['hasElevator'])
    hasBars = int(data['hasBars'])
    hasStorage = int(data['hasStorage'])
    hasAirCondition = int(data['hasAirCondition'])
    hasBalcony = int(data['hasBalcony'])
    handicapFriendly = int(data['handicapFriendly'])
    city_שוהם = int(data['City_שוהם'])
    city_אילת = int(data['City_אילת'])
    city_בית_שאן = int(data['City_ביתשאן'])
    city_בת_ים = int(data['City_בתים'])
    city_גבעת_שמואל = int(data['City_גבעתשמואל'])
    city_חולון = int(data['City_חולון'])
    city_ירושלים = int(data['City_ירושלים'])
    city_נהרייה = int(data['City_נהרייה'])
    city_נתניה = int(data['City_נתניה'])
    city_צפת = int(data['City_צפת'])
    city_קרית_ביאליק = int(data['City_קריתביאליק'])
    city_רחובות = int(data['City_רחובות'])
    city_תל_אביב = int(data['City_תלאביב'])
    type_אחר = int(data['type_אחר'])
    type_בית_פרטי = int(data['type_ביתפרטי'])
    type_בניין = int(data['type_בניין'])
    type_דו_משפחתי = int(data['type_דומשפחתי'])
    type_דופלקס = int(data['type_דופלקס'])
    type_דירה = int(data['type_דירה'])
    type_דירת_גג = int(data['type_דירתגג'])
    type_דירת_גן = int(data['type_דירתגן'])
    type_טריפלקס = int(data['type_טריפלקס'])
    type_מיני_פנטהאוז = int(data['type_מיניפנטהאוז'])
    type_נחלה = int(data['type_נחלה'])
    type_קוטג_טורי = int(data['type_קוטגטורי'])
    condition_None = int(data['condition_None'])
    condition_דורש_שיפוץ = int(data['condition_דורששיפוץ'])
    condition_חדש = int(data['condition_חדש'])
    condition_ישן = int(data['condition_ישן'])
    condition_משופץ = int(data['condition_משופץ'])
    condition_שמור = int(data['condition_שמור'])
    
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'room_number': [room_number],
        'area': [area],
        'hasElevator': [hasElevator],
        'hasBars': [hasBars],
        'hasStorage': [hasStorage],
        'hasAirCondition': [hasAirCondition],
        'hasBalcony': [hasBalcony],
        'handicapFriendly': [handicapFriendly],
        'city_שוהם': [city_שוהם],
        'city_אילת': [city_אילת],
        'city_בית_שאן': [city_בית_שאן],
        'city_בת_ים': [city_בת_ים],
        'city_גבעת_שמואל': [city_גבעת_שמואל],
        'city_חולון': [city_חולון],
        'city_ירושלים': [city_ירושלים],
        'city_נהרייה': [city_נהרייה],
        'city_נתניה': [city_נתניה],
        'city_צפת': [city_צפת],
        'city_קרית_ביאליק': [city_קרית_ביאליק],
        'city_רחובות': [city_רחובות],
        'city_תל_אביב': [city_תל_אביב],
        'type_אחר': [type_אחר],
        'type_בית_פרטי': [type_בית_פרטי],
        'type_בניין': [type_בניין],
        'type_דו_משפחתי': [type_דו_משפחתי],
        'type_דופלקס': [type_דופלקס],
        'type_דירה': [type_דירה],
        'type_דירת_גג': [type_דירת_גג],
        'type_דירת_גן': [type_דירת_גן],
        'type_טריפלקס': [type_טריפלקס],
        'type_מיני_פנטהאוז': [type_מיני_פנטהאוז],
        'type_נחלה': [type_נחלה],
        'type_קוטג_טורי': [type_קוטג_טורי],
        'condition_None': [condition_None],
        'condition_דורש_שיפוץ': [condition_דורש_שיפוץ],
        'condition_חדש': [condition_חדש],
        'condition_ישן': [condition_ישן],
        'condition_משופץ': [condition_משופץ],
        'condition_שמור': [condition_שמור]
    })
    


    # Perform prediction
    predicted_price = model.predict(input_data)[0]

    # Prepare the prediction response
    response = {
        'predicted_price': predicted_price
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)