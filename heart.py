import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("heart.csv")
x = df.drop(columns='target', axis=1)
y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)
    
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear',random_state = 0)
    svc_lin.fit(X_train,Y_train)
    
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf',random_state = 0)
    svc_rbf.fit(X_train,Y_train)
    
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train,Y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train,Y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train,Y_train)
    
    log.score(X_train,Y_train)
    knn.score(X_train,Y_train)
    svc_lin.score(X_train,Y_train)
    svc_rbf.score(X_train,Y_train)
    gauss.score(X_train,Y_train)
    tree.score(X_train,Y_train)
    forest.score(X_train,Y_train)
    return log,knn,svc_lin,svc_rbf,gauss,tree,forest


model = models(X_train, Y_train)


def heart():
    Age = int(input('Enter the age: '))
    sex = int(input('Enter the sex: '))
    cp = int(input('Chest pain (0 being the least to 3 being the highest): '))
    trestbps = int(input('Resting BP: '))
    chol = int(input('serum cholestoral in mg/dl: '))
    fbs = int(input('fasting blood sugar &gt; 120 mg/dl (1 = true; 0 = false): '))
    restecg = int(input('resting electrocardiographic results: '))
    thalach = int(input('maximum heart rate achieved: '))
    exang = int(input('exercise induced angina (1 = yes; 0 = no): '))
    oldpeak = float(input('ST depression induced by exercise relative to rest: '))
    slope = int(input('the slope of the peak exercise ST segment: '))
    ca = int(input('number of major vessels (0-3) colored by flourosopy: '))
    thal = int(input('1 = normal; 2 = fixed defect; 3 = reversable defect: '))
    query = pd.DataFrame({'age':[Age],'sex':[sex],'cp': [cp],'trestbps':[trestbps],'chol':[chol],'fbs':[fbs],'restecg':[restecg],'thalach':[thalach],'exang':[exang],'oldpeak':[oldpeak],'slope':[slope],'ca':[ca],'thal':[thal]})
    prediction = model[6].predict(query)
    print(prediction)
    if prediction[0] == 0:
        print('You do not have a Heart Disease')
    else:
        print('You have Heart Disease')
