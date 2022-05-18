import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv("hypertension.csv")
df['gender'] = df['gender'].replace({'Male':'0','Female':'1','Other':'2'})
df['ever_married'] = df['ever_married'].replace({'No':'0','Yes':'1'})
df['work_type'] = df['work_type'].replace({'Private':'0','Self-employed':'1','Govt_job':'2','children':'3','Never_worked':'4'})
df['Residence_type'] = df['Residence_type'].replace({'Urban':'0','Rural':'1'})
df['smoking_status'] = df['smoking_status'].replace({'formerly smoked':'0','never smoked':'1','smokes':'2','Unknown':'3'})
df.isnull().sum()
del df['bmi']
del df['id']
datatrue = len(df.loc[df['stroke'] == True])
datafalse = len(df.loc[df['stroke'] == False])
features = ['gender','age','stroke','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','smoking_status']
predicted = ['hypertension']
x = df[features].values
y = df[predicted].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
fill = SimpleImputer(missing_values=0,strategy="mean")
x_train = fill.fit_transform(x_train)
x_test = fill.fit_transform(x_test)


def models(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(x_train,y_train)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)
    knn.fit(x_train,y_train)
    
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear',random_state = 0)
    svc_lin.fit(x_train,y_train)
    
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf',random_state = 0)
    svc_rbf.fit(x_train,y_train)
    
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train,y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(x_train,y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(x_train,y_train)
    
    log.score(x_train,y_train)
    knn.score(x_train,y_train)
    svc_lin.score(x_train,y_train)
    svc_rbf.score(x_train,y_train)
    gauss.score(x_train,y_train)
    tree.score(x_train,y_train)
    forest.score(x_train,y_train)
    return log,knn,svc_lin,svc_rbf,gauss,tree,forest


model = models(x_train, y_train)


def hypertension():
    gender = int(input('Gender(Male=0, Female=1, Other=2): '))
    Age = int(input('Enter the age: '))
    stroke = int(input('stroke(no of strokes): '))
    heart_disease = int(input('Heart disease(yes=1, no=0): '))
    ever_married = int(input('Marriage status(yes=1, no=0): '))
    work_type = int(input('Work type(private=0, self-employed=1, govt job=2, children=3, never worked=4):'))
    Residence_type = int(input('Residence_type(urban=0, rural=1): '))
    avg_glucose_level = float(input('avg glucose level: '))
    smoking_status = int(input('smoking status(formerly somked=0, never smoked=1, smoker=2,unknown=3): '))
    query = pd.DataFrame({'gender':[gender],'age':[Age],'stroke':[stroke],'heart_disease':[heart_disease],'ever_married':[ever_married],'work_type':[work_type],'Residence_type':[Residence_type],'avg_glucose_level':[avg_glucose_level],'smoking_status':[smoking_status]})
    prediction = model[6].predict(query)
    print(prediction)
    if prediction[0] == 0:
        print('You do not have hypertension')
    else:
        print('You have Hypertension')
