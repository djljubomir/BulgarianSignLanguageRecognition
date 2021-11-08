from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np


sign_names = ["а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
              "х", "ц", "ч", "ш", "щ", "ъ", "ь", "ю", "я", "интервал", "изтриване", "нищо"]

X = np.load('sign_images.npy')
y = np.load('sign_indexes.npy')

print(X)

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)