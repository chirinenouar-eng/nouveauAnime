import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV # <--- NOUVEL IMPORT
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from fuzzywuzzy import process

# -----------------------------------------------------------------------------
# 1. Configuration de la page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anime Assistant & Analytics",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 🔐 Accès Restreint
# -----------------------------------------------------------------------------
# On vérifie d'abord si le mot de passe est déjà validé dans la session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Entrez le mot de passe pour accéder à l'application :", type="password")
    
    if password:
        if password == st.secrets["app_password"]:
            st.session_state.authenticated = True
            st.success("Accès autorisé ! Bienvenue.")
            st.rerun() # Recharge la page pour masquer le champ mot de passe
        else:
            st.error("Mot de passe incorrect. Veuillez réessayer.")
            st.stop()
    else:
        st.stop() # Attend que l'utilisateur tape quelque chose

# -----------------------------------------------------------------------------
# 2. Chargement des données et Entraînement (Mis en cache)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    # --- Chargement CSV ---
    try:
        df = pd.read_csv("anime.csv")
    except FileNotFoundError:
        return None, None, None

    # --- Nettoyage Dataframe ---
    cols = ["_id", "title", "genres", "synopsis", "ranking", "episodes", "type", "status", "image"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].copy()

    # Nettoyage des genres (string -> liste)
    def parse_genres(x):
        if isinstance(x, str):
            try:
                if x.strip().startswith("[") and x.endswith("]"):
                    return eval(x)
                return x.split(", ")
            except:
                return []
        return []

    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(parse_genres)

    # Nettoyage Ranking
    df["ranking"] = pd.to_numeric(df["ranking"], errors='coerce')
    
    # Suppression doublons et fillna
    df = df.drop_duplicates(subset="title")
    df["synopsis"] = df["synopsis"].fillna("No synopsis available.")
    
    # Gestion des images
    if "image" not in df.columns:
        df["image"] = "https://via.placeholder.com/225x318?text=No+Image"
    else:
        df["image"] = df["image"].fillna("https://via.placeholder.com/225x318?text=No+Image")

    # --- Données d'entraînement ---
    training_data = [
        # recommend_anime
        ("Je veux un anime à regarder", "recommend_anime"),
        ("Propose-moi un bon anime", "recommend_anime"),
        ("Donne-moi une recommandation", "recommend_anime"),
        ("Je cherche un anime sombre", "recommend_anime"),
        ("Un anime avec beaucoup d'action ?", "recommend_anime"),
        ("Je veux un anime psychologique", "recommend_anime"),
        ("Quel anime devrais-je regarder ce soir", "recommend_anime"),
        ("Un anime court à me conseiller ?", "recommend_anime"),
        ("Je veux un anime avec un héros tourmenté", "recommend_anime"),
        ("Trouve-moi un anime dans le style Berserk", "recommend_anime"),
        # recommend_by_genre
        ("Donne-moi un anime d’horreur", "recommend_by_genre"),
        ("Je veux un anime de science-fiction", "recommend_by_genre"),
        ("Un anime de comédie ?", "recommend_by_genre"),
        ("Je cherche un anime de fantasy", "recommend_by_genre"),
        ("Propose un anime romantique", "recommend_by_genre"),
        ("Un bon anime d’aventure ?", "recommend_by_genre"),
        ("Je veux un anime de sport", "recommend_by_genre"),
        # ask_info
        ("C'est quoi le synopsis de Akira ?", "ask_info"),
        ("Parle-moi de Berserk", "ask_info"),
        ("De quoi parle Mononoke ?", "ask_info"),
        ("synopsis de Perfect Blue", "ask_info"),
        ("Explique-moi l’histoire de Dorohedoro", "ask_info"),
        ("Quel est le pitch de Paprika ?", "ask_info"),
        # ask_genre
        ("Quel genre a Berserk ?", "ask_genre"),
        ("C’est quoi les genres de Akira ?", "ask_genre"),
        ("Quels sont les genres de Perfect Blue ?", "ask_genre"),
        ("Mononoke appartient à quel genre ?", "ask_genre"),
        # ask_ranking
        ("Quel est le ranking de Berserk ?", "ask_ranking"),
        ("C’est quoi la note de Perfect Blue ?", "ask_ranking"),
        ("Akira est bien classé ?", "ask_ranking"),
        ("Quel est l’anime le mieux noté ?", "ask_ranking"),
        # ask_episodes
        ("Combien d’épisodes a Berserk ?", "ask_episodes"),
        ("Perfect Blue dure combien de temps ?", "ask_episodes"),
        ("Cet anime est long ?", "ask_episodes"),
        ("Il y a combien d’épisodes dans Mononoke ?", "ask_episodes"),
        # ask_type
        ("Berserk c’est un film ou une série ?", "ask_type"),
        ("Akira c’est un long métrage ?", "ask_type"),
        ("Perfect Blue c’est une série ?", "ask_type"),
        # greeting
        ("Salut", "greeting"),
        ("Bonjour", "greeting"),
        ("Hey", "greeting"),
        ("Yo", "greeting"),
        ("Coucou", "greeting"),
        # goodbye
        ("Merci, au revoir", "goodbye"),
        ("A plus", "goodbye"),
        ("Bonne soirée", "goodbye"),
        ("Ciao", "goodbye"),
        ("On se reparle plus tard", "goodbye"),
        ("Au revoir", "goodbye"),
        # fallback
        ("Je ne comprends pas", "fallback"),
        ("Hein ?", "fallback"),
        ("???", "fallback"),
        ("Je suis perdu", "fallback"),
    ]

    train_df = pd.DataFrame(training_data, columns=["text", "intent"])

    # --- Entraînement Pipeline (Modifié pour probabilités) ---
    svc = LinearSVC(dual="auto", random_state=42)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", CalibratedClassifierCV(svc)) # Permet d'avoir des pourcentages de confiance
    ])
    clf.fit(train_df["text"], train_df["intent"])
    
    return df, clf, train_df

# Chargement
df, clf, train_df = load_resources()

# -----------------------------------------------------------------------------
# 3. Fonctions Logiques (Chatbot)
# -----------------------------------------------------------------------------

def extract_title(text, dataframe, min_score=70):
    if dataframe is None: return None
    cleaned = text.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    titles = dataframe["title"].dropna().tolist()
    words = cleaned.split()
    candidates = []
    for i in range(len(words)):
        for j in range(i+1, min(i+5, len(words))+1):
            candidates.append(" ".join(words[i:j]))
            
    best_match = None
    best_score = 0
    for candidate in candidates:
        match, score = process.extractOne(candidate, titles)
        if score > best_score and score >= min_score:
            best_match = match
            best_score = score
    return best_match

def get_bot_response(user_input, dataframe, model):
    """
    Retourne (réponse_texte, dataframe_confiance)
    """
    if dataframe is None: return "Erreur : Base de données non chargée.", None
    
    user_input_clean = user_input.rstrip("!?.,;: ").strip()
    if not user_input_clean: return "Je n’ai rien compris, peux-tu reformuler ?", None
    
    # 1. Prédiction des probabilités pour CHAQUE intention
    proba = model.predict_proba([user_input_clean])[0]
    classes = model.classes_
    
    # 2. Création du tableau de confiance
    conf_data = pd.DataFrame({
        "Intention": classes,
        "Confiance": proba
    }).sort_values(by="Confiance", ascending=False)
    
    # 3. L'intention gagnante
    intent = conf_data.iloc[0]["Intention"]
    
    # --- Logique de réponse ---
    response = "Désolé, je n'ai pas compris."

    # Greeting / Goodbye / Fallback
    if intent == "greeting":
        response = "Salut ! Je suis ton assistant Anime. Tu peux me demander des recommandations ou des infos sur un titre."
    elif intent == "goodbye":
        response = "À bientôt ! Reviens quand tu veux."
    elif intent == "fallback":
        response = "Je ne suis pas sûr de comprendre. Essaie de demander 'Recommande un anime d'action' ou 'Synopsis de Akira'."

    # Extraction de titre pour les questions spécifiques
    title = extract_title(user_input_clean, dataframe)
    
    if intent in ["ask_info", "ask_genre", "ask_type", "ask_episodes", "ask_ranking"]:
        if not title:
            response = "Je n'ai pas trouvé de nom d'anime dans ta phrase. Peux-tu préciser ?"
        else:
            row = dataframe[dataframe["title"].str.contains(title, case=False, regex=False)]
            if len(row) == 0: 
                response = "Je ne trouve pas cet anime dans ma base."
            else:
                row = row.iloc[0]
                if intent == "ask_info":
                    response = f"📺 **{row['title']}**\n\n{row['synopsis']}"
                elif intent == "ask_genre":
                    genres_str = ", ".join(row['genres']) if isinstance(row['genres'], list) else str(row['genres'])
                    response = f"Les genres de **{row['title']}** sont : {genres_str}"
                elif intent == "ask_type":
                    response = f"**{row['title']}** est un(e) **{row['type']}**."
                elif intent == "ask_episodes":
                    response = f"**{row['title']}** contient **{row['episodes']}** épisode(s)."
                elif intent == "ask_ranking":
                    response = f"**{row['title']}** est classé top **{row['ranking']}** sur MyAnimeList."

    # Recommendation Logic
    if intent == "recommend_anime":
        genres_list = ["action", "horror", "fantasy", "drama", "comedy", "romance", "sci-fi"]
        found = [g for g in genres_list if g in user_input_clean.lower()]
        if found:
            g_req = found[0].capitalize()
            subset = dataframe[dataframe["genres"].apply(lambda x: g_req in x if isinstance(x, list) else False)]
            if not subset.empty:
                anime = subset.sample(1).iloc[0]
                response = f"Je te recommande **{anime['title']}** (Genre : {g_req})."
            else:
                response = f"Je n'ai rien trouvé en {g_req}."
        else:
            anime = dataframe.sample(1).iloc[0]
            response = f"Je te recommande au hasard : **{anime['title']}**."

    if intent == "recommend_by_genre":
        all_genres = ["Action", "Horror", "Fantasy", "Drama", "Comedy", "Romance", "Sci-Fi", "Adventure", "Mystery", "Sports", "Suspense", "Slice of Life"]
        found = [g for g in all_genres if g.lower() in user_input_clean.lower()]
        if found:
            g_req = found[0]
            subset = dataframe[dataframe["genres"].apply(lambda x: g_req in x if isinstance(x, list) else False)]
            if not subset.empty:
                anime = subset.sample(1).iloc[0]
                response = f"Voici une pépite en {g_req} : **{anime['title']}**."
            else:
                response = f"Désolé, je n'ai rien trouvé en {g_req}."
        else:
            response = "Quel genre cherches-tu ? (Action, Horror, Comedy...)"

    return response, conf_data

# -----------------------------------------------------------------------------
# 4. Interface Streamlit (Layout Principal)
# -----------------------------------------------------------------------------

if df is None:
    st.error("Le fichier 'anime.csv' est manquant. Veuillez l'ajouter au dossier.")
else:
    
    # --- SIDEBAR AVEC IMAGES ---
    with st.sidebar:
        # Bouton déconnexion
        if st.button("Se déconnecter"):
            st.session_state.authenticated = False
            st.rerun()

        st.header("🖼️ Galerie & Info")
        
        # Section 1 : Anime du moment
        st.subheader("🏆 Anime du moment")
        top_anime = df.sort_values(by="ranking", ascending=True).iloc[0]
        
        st.image(top_anime["image"], caption=f"Top #1 : {top_anime['title']}", use_container_width=True)
        with st.expander("Voir le synopsis"):
            st.write(top_anime["synopsis"][:300] + "...")

        st.markdown("---")

        # Section 2 : Découverte Aléatoire
        st.subheader("🎲 Découverte")
        if st.button("Surprends-moi !"):
            random_anime = df.sample(1).iloc[0]
            st.session_state["random_anime"] = random_anime
        
        if "random_anime" in st.session_state:
            rand_anim = st.session_state["random_anime"]
            st.image(rand_anim["image"], caption=rand_anim["title"], use_container_width=True)
            st.write(f"**Genre:** {', '.join(rand_anim['genres']) if isinstance(rand_anim['genres'], list) else rand_anim['genres']}")


    # --- CONTENU PRINCIPAL ---
    st.title("🎌 Anime Assistant & Analytics")

    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📊 Exploration Données", "⚙️ Performance Modèle"])

    # -------------------------------------------------------------------------
    # ONGLET 1 : CHATBOT
    # -------------------------------------------------------------------------
    with tab1:
        st.header("🤖 Discute avec le Bot")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Salut ! Pose-moi une question sur un anime ou demande une recommandation."}]

        # Affichage historique
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input utilisateur
        if prompt := st.chat_input("Votre message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Analyse..."):
                # Récupération réponse ET confiance
                resp, conf_data = get_bot_response(prompt, df, clf)
            
            st.session_state.messages.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)
                
                # --- VISUALISATION DE LA CONFIANCE ---
                if conf_data is not None:
                    with st.expander("🧠 Voir la confiance du modèle"):
                        # Graphique
                        fig_conf = px.bar(
                            conf_data, 
                            x="Confiance", 
                            y="Intention", 
                            orientation='h',
                            text_auto='.1%', # Format pourcentage
                            title="Probabilité par intention",
                            color="Confiance",
                            color_continuous_scale="Blues"
                        )
                        fig_conf.update_layout(yaxis={'categoryorder':'total ascending'}, height=300)
                        st.plotly_chart(fig_conf, use_container_width=True)
                        
                        # Info textuelle
                        top = conf_data.iloc[0]
                        st.caption(f"Intention détectée : **{top['Intention']}** avec **{top['Confiance']*100:.1f}%** de certitude.")

    # -------------------------------------------------------------------------
    # ONGLET 2 : EXPLORATION (METRIQUES)
    # -------------------------------------------------------------------------
    with tab2:
        st.header("📊 Statistiques des Animes")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Animes", len(df))
        col2.metric("Meilleur Rang", df["ranking"].min() if "ranking" in df else "N/A")
        col3.metric("Moyenne Épisodes", f"{df['episodes'].replace('Unknown', 0).astype(float).mean():.1f}")
        col4.metric("Types différents", df["type"].nunique())

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            type_counts = df["type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            fig_type = px.pie(type_counts, values="Count", names="Type", title="Distribution par Type", hole=0.4)
            st.plotly_chart(fig_type, use_container_width=True)

        with col_g2:
            all_genres = df.explode("genres")
            genre_counts = all_genres["genres"].value_counts().head(10).reset_index()
            genre_counts.columns = ["Genre", "Count"]
            fig_genre = px.bar(genre_counts, x="Count", y="Genre", orientation='h', 
                               title="Top 10 Genres les plus fréquents", color="Count", color_continuous_scale="Viridis")
            fig_genre.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_genre, use_container_width=True)

        st.markdown("---")
        
        st.subheader("🏆 Top 10 Animes les mieux notés")
        top_animes = df.sort_values(by="ranking", ascending=True).head(10)[["ranking", "title", "type", "episodes", "status"]]
        st.dataframe(top_animes, use_container_width=True)

    # -------------------------------------------------------------------------
    # ONGLET 3 : PERFORMANCE MODELE
    # -------------------------------------------------------------------------
    with tab3:
        st.header("⚙️ Performance du modèle de classification (SVM)")
        
        st.info("Le modèle est entraîné sur un petit jeu de données de phrases types (intentions).")

        # Prédictions sur le jeu d'entraînement pour calculer les métriques
        y_true = train_df["intent"]
        y_pred = clf.predict(train_df["text"])
        labels = sorted(list(set(y_true)))

        col_p1, col_p2 = st.columns([2, 1])

        with col_p1:
            st.subheader("📈 Matrice de Confusion")
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # Heatmap Plotly
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Prédiction", y="Vraie Intention", color="Nombre"),
                x=labels,
                y=labels,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=600)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_p2:
            st.subheader("📋 Rapport de Classification")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🧠 Analyse des données d'entraînement")
        
        intent_counts = train_df["intent"].value_counts().reset_index()
        intent_counts.columns = ["Intention", "Nombre d'exemples"]
        
        fig_intent = px.bar(intent_counts, x="Intention", y="Nombre d'exemples", 
                            title="Distribution des classes (Intentions)",
                            color="Nombre d'exemples", color_continuous_scale="Reds")
        st.plotly_chart(fig_intent, use_container_width=True)