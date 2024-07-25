# Recommandation-et-prédiction-de-films
Ce projet développe un système de recommandation et de prédiction de films basé sur les données MovieLens (20M ratings, 138K utilisateurs, 1995-2015) disponibles sur Kaggle . 
# Dataset
- Source MovieLens: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
- 20 000 263 ratings de films, et 138493 utilisateurs 
- Entre janvier 1995 et mars 2015
- Le nombre de ratings ou utilisateurs utilisés est paramétrable.
- Les colonnes utilisés : 
 Rating.csv : userId, movieId, rating
 Movie.csv : movieId, title, genres
# Algo /Modèle
- Etape 1: Création des TF-IDF sur les titres des films et sur leurs catégories.
- Etape 2: Modèle Item Based (IB) Filtering, Pour chaque film: rating = 95% x (IB sur Catégorie ) + 5% x (IB sur Titre) 
- Etape 3: Modèle User Based Filtering
- Etape 4: Fusion des modèles pour chaque film selon un ratio paramétrable (par défaut 70% UB, 30%IB). (Prédiction) 
- Etape 5: Afficher les films par ordre du rating le plus élevé d’abord jusqu’au plus bas. ( Recommandation)
