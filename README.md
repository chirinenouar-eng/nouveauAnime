🎌 Anime Assistant & Analytics
📊 Description Ce projet est une application web interactive conçue pour les passionnés d'animation japonaise. Elle combine la puissance du Natural Language Processing (NLP) et de la Data Visualization pour offrir une expérience utilisateur fluide et informative.

L'objectif est double : permettre aux utilisateurs d'explorer une vaste base de données d'animes via des graphiques dynamiques, tout en interagissant avec un agent conversationnel intelligent capable de comprendre des requêtes naturelles, d'extraire des informations spécifiques et de recommander du contenu.

L'application met un point d'honneur sur la transparence de l'IA en permettant d'analyser en temps réel comment le modèle prend ses décisions et où se situent ses incertitudes.

🎯 Parcours Parcours B : Projet Personnel sur la recommandation et l'analyse d'Animes.

📁 Dataset

Source : MyAnimeList Dataset (Kaggle) ou extraction CSV locale.

Taille : +10 000 lignes, 12 colonnes.

Variables principales : title, synopsis, genres, ranking, episodes, type.

Variable cible : intent (utilisée pour l'entraînement du chatbot).

🚀 Fonctionnalités

Page 1 : 🤖 Chatbot Intelligent
Dialogue Naturel : Posez des questions sur les synopsis, les genres ou demandez des recommandations.

Reconnaissance d'intention : Pipeline NLP (TF-IDF + SVM) pour classifier les requêtes.

Recherche Floue : Intégration de FuzzyWuzzy pour gérer les fautes de frappe sur les titres (ex: "Berzerk" ➔ "Berserk").

Nouveau : Matrice de Confiance : Pour chaque réponse, le bot affiche désormais un score de probabilité. Si le modèle hésite entre deux intentions (ex: "recommandation" vs "recherche info"), un graphique de confiance montre les scores comparatifs des différents intents.

Page 2 : 📊 Exploration des Données
KPIs Flash : Compteur total, score moyen, et distribution des formats.

Visualisations Plotly : * Répartition par type (TV, Movie, OVA) via Pie Chart.

Top des genres les plus représentés via Bar Chart.

Exploration filtrée : Tableau interactif du Top 10 selon les préférences.

Page 3 : ⚙️ Performance du Modèle
Matrice de Confusion : Visualisation globale des erreurs de classification du modèle sur le set de test.

Classification Report : Détail de la Précision et du Rappel pour chaque intention.

🖼️ Sidebar (Barre latérale)
Affiche l'affiche de l'anime numéro 1 du classement actuel.

Bouton "Surprends-moi !" : Génère une fiche aléatoire avec image et résumé.

🛠️ Technologies Utilisées

Python 3.8+

Streamlit : Interface utilisateur.

Pandas : Manipulation des données.

Scikit-learn : Entraînement du modèle SVM et Vectorisation.

Plotly Express : Graphiques interactifs.

FuzzyWuzzy : Matching de chaînes de caractères.

📦 Installation Locale

Bash
# 1. Cloner le repository
git clone https://github.com/votre-username/anime-assistant.git
cd anime-assistant

# 2. Créer un environnement virtuel
python -m venv .venv
# Windows : .venv\Scripts\activate | Mac/Linux : source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
🌐 Déploiement Application déployée sur Streamlit Cloud : 👉(https://nouveauanime-jerbnhgzup99aa7q4t3l2o.streamlit.app/)

👥 Équipe

[Votre Nom] - Développeur Fullstack & Data Scientist

📝 Notes

Défi technique : L'implémentation de la matrice de confiance a nécessité de passer d'un LinearSVC (qui ne gère pas nativement les probabilités) à un modèle capable d'utiliser predict_proba.

Améliorations futures : Intégration d'un système de recommandation basé sur le filtrage collaboratif (User-based).