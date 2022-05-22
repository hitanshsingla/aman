import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def get_forest():
    df = pd.read_csv("diabetes.csv")
    features = ['Glucose', 'Age', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'SkinThickness',
                'Pregnancies']
    predicted = ['Outcome']

    x = df[features].values
    y = df[predicted].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    fill = SimpleImputer(missing_values=0, strategy="mean")
    x_train = fill.fit_transform(x_train)

    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)
    return forest
    # Decision tree has over-fitting.


def diabetes(glucose, age, bloodpressure, insulin, bmi, diabetespedigreefunction, skinthickness, pregnancies):
    query = pd.DataFrame({
        'Glucose': [int(glucose)],
        'Age': [int(age)],
        'BloodPressure': [int(bloodpressure)],
        'Insulin': [int(insulin)],
        'BMI': [float(bmi)],
        'DiabetesPedigreeFunction': [float(diabetespedigreefunction)],
        'SkinThickness': [int(skinthickness)],
        'Pregnancies': [int(pregnancies)]})

    model = get_forest()
    prediction = model[1].predict(query)
    # print(prediction)
    print("With accuracy level of 75% ")
    if prediction[0] == 0:
        print("The Person does not have a Diabetes")
    else:
        print("The Person has Diabetes")


diabetes(148, 50, 72, 0, 33, 0.6, 35, 6)
