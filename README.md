# 🎌 Anime Assistant & Analytics
 
Bienvenue sur le projet **Anime Assistant**. Il s'agit d'une application web interactive développée avec **Streamlit** qui combine un chatbot de recommandation (basé sur du Machine Learning) et un tableau de bord analytique pour explorer une base de données d'animes.
 
---
 
## 📋 Fonctionnalités
 
L'application est organisée en **3 onglets principaux** et une **barre latérale** :
 
### 1. 🤖 Chatbot Intelligent
* **Dialogue Naturel** : Posez des questions comme *"Je cherche un anime d'action"* ou *"Synopsis de Akira"*.
* **Reconnaissance d'intention** : Un modèle NLP (TF-IDF + SVM) identifie si vous dites bonjour, cherchez une info, ou voulez une recommandation.
* **Tolérance aux fautes** : Grâce à l'algorithme de distance de Levenshtein (FuzzyWuzzy), l'application comprend *"Berzerk"* même si le titre réel est *"Berserk"*.
 
### 2. 📊 Exploration des Données
* **KPIs en temps réel** : Nombre total d'animes, moyenne d'épisodes, types différents.
* **Visualisation Interactive** :
    * Graphiques circulaires (Pie charts) pour la répartition TV/Films/OAV.
    * Diagrammes en barres pour les genres les plus populaires.
* **Classement** : Tableau dynamique du Top 10 des animes les mieux notés.
 
### 3. ⚙️ Performance du Modèle
* Une vue transparente sur le "cerveau" du chatbot.
* Affichage de la **Matrice de Confusion** pour voir où le bot pourrait se tromper.
* Rapport de classification (Précision, Rappel) sur les données d'entraînement.
 
### 🖼️ Barre Latérale (Sidebar)
* Affiche l'affiche de l'anime #1 du classement.
* Bouton **"Surprends-moi !"** pour découvrir un anime aléatoire avec son image.
 
---
 
## 🛠️ Prérequis
 
Avant de commencer, assurez-vous d'avoir installé :
* **Python** (version 3.8 ou supérieure).
* **Git** (optionnel, pour cloner le projet).
 
---
 
## 📦 Installation (Étape par étape)
 
Suivez ces étapes pour lancer le projet sur votre machine locale.
 
### Étape 1 : Récupérer le projet
Si vous avez Git :
```bash
git clone [https://github.com/votre-username/anime-assistant.git](https://github.com/votre-username/anime-assistant.git)
cd anime-assistant
```
### Étape 2 : Créer un environnement virtuel
Sous Windows :
```bash
python -m venv .venv
.venv\Scripts\activate
```

Sous Mac/Linux :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3 : Installer les dépendances
Nous utilisons le fichier requirements.txt pour installer Streamlit, Pandas, Scikit-learn, etc.
```bash
pip install -r requirements.txt
```

### Étape 4 : Vérifier les données
Assurez-vous que le fichier anime.csv est bien présent à la racine du dossier (au même endroit que app.py). Ce fichier doit contenir au minimum les colonnes : title, synopsis, genres, ranking, episodes, type, image.

## 🚀 Lancement de l'application
Une fois l'application terminée, lancez l'application avec la commande suivante :
```bash
streamlit run app.py
```
Votre navigateur va s'ouvrir automatiquement à l'adresse : http://localhost:8501

## 🧠 Comment ça marche techniquement ?
Le chatbot n'utilise pas d'API externe coûteuse (comme OpenAI). Il fonctionne en local grâce à une pipeline Scikit-learn : 
1. Entraînement à la volée : Au lancement de l'app, le script entraîne un modèle sur une petite liste de phrases types (intentions) définies dans le code.

2. Vectorisation : Le texte utilisateur est transformé en vecteurs numériques via TfidfVectorizer.

3. Classification : Un classifieur LinearSVC prédit l'intention (ex: ask_genre).

4. Extraction d'entité : Si l'intention nécessite un titre (ex: "Genre de Naruto"), fuzzywuzzy cherche le titre le plus proche dans le fichier CSV

## 📂 Structure du Projet
anime-assistant/
├── .gitignore          # Fichiers à ignorer par Git (venv, cache, secrets)
├── app.py              # Le code principal de l'application Streamlit
├── anime.csv           # La base de données (source)
├── requirements.txt    # Liste des librairies Python nécessaires
└── README.md           # Ce fichier de documentation

---

### 👤 Auteur
Projet réalisé par **Chirine Nouar & Glenn Mboga**.

