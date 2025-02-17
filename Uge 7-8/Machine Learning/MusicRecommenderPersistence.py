import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Tidligere træning af modellen (udkommenteret, da vi nu bruger en gemt model)
# Læser musikdata fra CSV-fil
# music_data = pd.read_csv(r'C:\Users\rasmu\OneDrive\Skrivebord\Produkter4.sem\Uge 7-8\Machine Learning\music.csv')  

# Opdeler data i input (X) og output (y)
# X = music_data.drop(columns=['genre'])  # Features (alder, køn)
# y = music_data['genre']  # Labels (musikgenre)

# Opretter og træner modellen
# model = DecisionTreeClassifier()
# model.fit(X, y)  # Træner modellen med X som input og y som output

# Gemmer den trænede model i en fil for senere brug
# joblib.dump(model, 'music-recommender.joblib')

# Indlæser den gemte model fra fil
model = joblib.load('music-recommender.joblib')

# Forudsiger musikgenre baseret på alder og køn (21 år, mand)
predictions = model.predict([[21, 1]])

# Udskriver den forudsagte genre
print(predictions)
