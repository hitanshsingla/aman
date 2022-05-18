import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('obesity.csv')
df.head()
df['Gender'] = df['Gender'].replace({'Male':'0','Female':'1'})
df.isnull().sum()
features = ['Gender','Height','Weight']
predicted = ['Index']
x = df[features].values
y = df[predicted].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)


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


def obesity():
    gender = int(input('Gender(Male=0, Female=1): '))
    h = int(input('press 1 for height in feet and inches, press 2 for height in cm'))
    if h == 1:
        f = int(input('feet: '))
        i = int(input('inches: '))
        height = (f*30.48)+(i*2.54)
    elif h == 2:
        height = int(input('Height in cm: '))
    weight = int(input('Weight in kg: '))
    query = pd.DataFrame({'Gender':[gender],'Height':[height],'Weight':[weight]})
    prediction = model[6].predict(query)
    print(prediction)
    if prediction[0] == 0:
        print('You are extremely weak')
    elif prediction[0] == 1:
        print('You are weak')
    elif prediction[0] == 2:
        print('You are normal')
    elif prediction[0] == 3:
        print('You are overweight')
    elif prediction[0] == 4:
        print('You are at risk of obesity')
    elif prediction[0] == 5:
        print('You are at risk of extreme obesity')
