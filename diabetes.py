import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")
datatrue = len(df.loc[df['Outcome'] == True])
datafalse = len(df.loc[df['Outcome'] == False])
features = ['Glucose', 'Age', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'SkinThickness',
            'Pregnancies']
predicted = ['Outcome']
x = df[features].values
y = df[predicted].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
fill = SimpleImputer(missing_values=0, strategy="mean")
x_train = fill.fit_transform(x_train)
x_test = fill.fit_transform(x_test)


def models(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(x_train, y_train)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(x_train, y_train)

    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(x_train, y_train)

    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(x_train, y_train)

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    log.score(x_train, y_train)
    knn.score(x_train, y_train)
    svc_lin.score(x_train, y_train)
    svc_rbf.score(x_train, y_train)
    gauss.score(x_train, y_train)
    tree.score(x_train, y_train)
    forest.score(x_train, y_train)
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest
# Decision tree has over-fitting.

model = models(x_train, y_train)


def diabetes():
    glucose = int(input('Enter the Glucose level: '))
    age = int(input('Enter the age: '))
    bloodpressure = int(input('Blood Pressure: '))
    insulin = int(input('Insulin: '))
    bmi = float(input('BMI: '))
    diabetespedigreefunction = float(input('DiabetesPedigreeFunction:'))
    skinthickness = int(input('SkinThickness'))
    pregnancies = int(input('Pregnancies'))
    query = pd.DataFrame(
        {'Glucose': [glucose], 'Age': [age], 'BloodPressure': [bloodpressure], 'Insulin': [insulin], 'BMI': [bmi],
         'DiabetesPedigreeFunction': [diabetespedigreefunction], 'SkinThickness': [skinthickness],
         'Pregnancies': [pregnancies]})
    prediction = model[5].predict(query)
    print(prediction)
    if prediction[0] == 0:
        print("The Person does not have a Diabetes")
    else:
        print("The Person has Diabetes")

diabetes()