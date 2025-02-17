import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Læser musikdata fra CSV-fil
music_data = pd.read_csv(r'C:\Users\rasmu\OneDrive\Skrivebord\Produkter4.sem\Uge 7-8\Machine Learning\music.csv')

# Opdeler data i input (X) og output (y)
X = music_data.drop(columns=['genre'])  # Features (alder, køn)
y = music_data['genre']  # Labels (musikgenre)

# Opdeler data i trænings- og test-sæt (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
# Jo mere data til træning, desto højere præcision, men risiko for overfitting

# Opretter og træner modellen
model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # Træner modellen med træningsdata

# Forudsiger musikgenre for testdata
predictions = model.predict(X_test)  

# Beregner modellens præcision ved at sammenligne forudsigelser med rigtige værdier
score = accuracy_score(y_test, predictions)  

# Udskriver præcision og forudsigelser
print("Model accuracy:", score)
print("Predictions:", predictions)
