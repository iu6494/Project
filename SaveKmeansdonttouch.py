import pandas as pd  # Importiert Pandas-Bibliothek und nennt sie pd
import numpy as np  # Importiert NumPy-Bibliothek und nennt sie np
import nltk  # Importiert NLTK-Bibliothek
from nltk.corpus import stopwords  # Importiert NLTK -Stoppwortliste
from nltk.tokenize import word_tokenize  # Importiert NLTK - Tokenisierer
from nltk.stem import WordNetLemmatizer  # Importiert NLTK - Lemmatizer
from sklearn.cluster import KMeans  # Importiert Scikit-Learn - KMeans-Modell
from sklearn.decomposition import LatentDirichletAllocation  # Importiert Scikit-Learn - LDA-Modell
from sklearn.feature_extraction.text import TfidfVectorizer  # Importiert Scikit-Learn - TF-IDF-Vektorisierer
from sklearn.metrics import davies_bouldin_score  # Importiert Scikit-Learn - Davies-Bouldin Score
from sklearn.metrics.pairwise import cosine_similarity  # Importiert Scikit-Learn - Kosinus-Ähnlichkeitsmetrik

# Lädt Englische Stoppwortliste
stopwortliste = stopwords.words('english')

# Lädt CSV-Datei und füllt NaN-Werte mit leerem String auf
reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_daten['Reports'].tolist()

# Wandelt jede complaint (einzelne Beschwerde) in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports (Sammlung aller lemmatisierten Beschwerden) hinzu
lemmatisierer = WordNetLemmatizer()  # Initialisierung WordNetLemmatizer als Variable
lemmatisierte_reports = []  # Erstellt leere Liste mit Variablen Namen lemmatisierte_reports
for complaint in reports:  # for-Schleife die über jede einzelne Beschwerde in der Liste reports (Sammlung aller Beschwerden) iteriert
    tokens = word_tokenize(complaint.lower())  # Tokenisiert jede einzelne Beschwerden und wandelt sie in Kleinbuchstaben um
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste]  # Lemmatisiert Tokens (running => run) und speichert die Lemmas, insofern es sich dabei nicht um ein Wort in der Stoppwortliste handelt, in der Variable/Liste lemmatisierte_tokens
    lemmatisierte_complaint = ' '.join(lemmatisierte_tokens)  # Fügt lemmatisierte Tokens zu einem String zusammen (' '.join fügt Leerzeichen ein)
    lemmatisierte_reports.append(lemmatisierte_complaint)  # Fügt lemmatisierten Beschwerdetext zur Liste hinzu

# TF-IDF erstellen
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Initialisierung des TF-IDF-Vektorisierers, mit engl. Stoppwortliste und Bigrammen
tfidf_matrix = tfidf_vectorizer.fit_transform(lemmatisierte_reports)  # Anwendung des TF-IDF-Vektorisierers auf lemmatisierte Beschwerdetexte (fit_transform ermöglicht Vektordarstellung)

# Führt KMeans-Clustering durch
num_cluster = 10  # Anzahl der Cluster
kmeans = KMeans(num_cluster, random_state=1)  # Initialisierung KMeans-Modell /Erstellt KMeans-Objekt (mit random_state für reprodzierbare Ergebnisse) aus der Scikit-learn Bibliothek
kmeans.fit(tfidf_matrix)  # Passt KMeans-Modell an die übergebene TF-IDF-Matrix an (s. Codezeile 31)

# weist jeder einzelnen Beschwerde aus der CSV-Datei eine Nummer zu die ein Thema repräsentiert
cluster_labels = kmeans.labels_

# Berechnung des Davies-Bouldin Index
db_index = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)  # Berechnet den Davies-Bouldin Index für die Cluster
best_db_index = float('inf')  # Initialisiert den besten Davies-Bouldin Index als unendlich
best_num_clusters = 15  # Initialisiert die beste Anzahl von Clustern mit einem Startwert

min_clusters = 5  # Minimale Anzahl von Clustern
max_clusters = 20  # Maximale Anzahl von Clustern

# Iteration über verschiedene Cluster-Anzahlen
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=1)  # Initialisierung des KMeans-Modells
    kmeans.fit(tfidf_matrix)  # Anpassung des KMeans-Modells an die Daten
    cluster_labels = kmeans.labels_  # Zuordnung jedes Datensatzes zu einem Cluster
    db_index = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)  # Berechnung des Davies-Bouldin Index für die aktuellen Cluster
    if db_index < best_db_index:  # Überprüfung, ob der aktuelle Index besser ist als der bisher beste
        best_db_index = db_index  # Aktualisierung des besten Davies-Bouldin Index
        best_num_clusters = num_clusters  # Aktualisierung der Anzahl der besten Cluster

# Ausgabe der gruppierten Beschwerden für jedes Cluster
for cluster_idx in range(best_num_clusters):
    print("Cluster " + str(cluster_idx) + ":")
    print()
    for idx, complaint in enumerate(reports):
        if cluster_labels[idx] == cluster_idx:  # Überprüfung, ob die Beschwerde zu diesem Cluster gehört
            print(complaint)  # Ausgabe der Beschwerde
            print()

print("Davies-Bouldin Index:", db_index)  # Ausgabe des Davies-Bouldin Index
print("Beste Anzahl von Clustern:", best_num_clusters)  # Ausgabe der besten Anzahl von Clustern
print("Davies-Bouldin Index für die beste Anzahl von Clustern:", best_db_index)  # Ausgabe des besten Davies-Bouldin Index
print()

# Initialisierung des CountVectorizer für die Term-Frequency-Inverse-Document-Frequency (TF-IDF)-Vektorisierung
DTM_Vectorizer = TfidfVectorizer(stop_words='english')  # Initialisiert den Vectorizer mit der englischen Stoppwortliste

# Berechnung des UMass-Coherence-Scores für ein Latent Dirichlet Allocation (LDA)-Modell
def umass_coherence(lda_model, dtm_cluster):
    coherence_score = 0.0  # Initialisiert den Kohärenzscore
    num_topics = lda_model.n_components  # Anzahl der Themen im LDA-Modell
    topic_word_matrix = lda_model.components_  # Matrix der Themenwörter
    vocabulary_size = dtm_cluster.shape[1]  # Größe des Vokabulars
    feature_names = list(DTM_Vectorizer.get_feature_names_out())  # Liste der Feature-Namen im Vokabular

    for topic_idx, topic in enumerate(topic_word_matrix):  # Iteriert über jedes Thema im LDA-Modell
        top_words_idx = topic.argsort()[:-3 - 1:-1]  # Indizes der Top-Wörter im Thema
        top_words = [feature_names[i] for i in top_words_idx]  # Extrahiert die Top-Wörter

        pairwise_coherence = 0.0  # Initialisiert die paarweise Kohärenz
        for i in range(len(top_words)):  # Iteriert über jedes Top-Wort
            for j in range(i+1, len(top_words)):  # Iteriert über die restlichen Top-Wörter
                word1_idx = feature_names.index(top_words[i])  # Index des ersten Worts
                word2_idx = feature_names.index(top_words[j])  # Index des zweiten Worts
                word_pair_similarity = cosine_similarity(topic[word1_idx].reshape(1, -1), topic[word2_idx].reshape(1, -1))  # Berechnet die Ähnlichkeit zwischen den Wortpaaren
                pairwise_coherence += word_pair_similarity  # Summiert die Ähnlichkeiten auf

        coherence_score += pairwise_coherence / (len(top_words) * (len(top_words) - 1) / 2)  # Berechnet den Kohärenzscore für das Thema

    return coherence_score / num_topics  # Gibt den durchschnittlichen Kohärenzscore über alle Themen zurück

# Initialisierung und Anpassung des LDA-Modells
DTM_Cluster = DTM_Vectorizer.fit_transform(lemmatisierte_reports)  # Anwendung des Vectorizers auf die lemmatisierten Berichte
lda = LatentDirichletAllocation(best_num_clusters, random_state=1)  # Initialisierung des LDA-Modells mit der besten Anzahl von Clustern
lda.fit(DTM_Cluster)  # Anpassung des LDA-Modells an die Daten

# Berechnung des UMass-Coherence-Scores für das LDA-Modell
umass_score = umass_coherence(lda, DTM_Cluster)  # Berechnet den Kohärenzscore
print("UMass Coherence Score:", umass_score)  # Gibt den UMass-Kohärenzscore aus
print()

# Ausgabe der Top-Wörter für jedes Cluster
for cluster_index in range(best_num_clusters):  # Iteriert über jedes Cluster
    print("Cluster " + str(cluster_index) + ":")  # Gibt die Cluster-Nummer aus
    print()
    valid_documents = [lemmatisierte_reports[idx] for idx in range(len(reports)) if cluster_labels[idx] == cluster_index if len(reports[idx].split()) > 2]  # Extrahiert die gültigen Dokumente für dieses Cluster
    if valid_documents:  # Überprüft, ob gültige Dokumente vorhanden sind
        DTM_Vectorizer = TfidfVectorizer(stop_words='english')  # Initialisierung des Vectorizers für die gültigen Dokumente
        DTM_Cluster = DTM_Vectorizer.fit_transform(valid_documents)  # Anwendung des Vectorizers auf die gültigen Dokumente
        lda = LatentDirichletAllocation(1, random_state=1)  # Initialisierung eines LDA-Modells mit einem Thema
        lda.fit(DTM_Cluster)  # Anpassung des LDA-Modells an die gültigen Dokumente
        feature_names = DTM_Vectorizer.get_feature_names_out()  # Extrahiert die Feature-Namen
        top_words_idx = lda.components_[0].argsort()[:-3 - 1:-1]  # Indizes der Top-Wörter im Thema
        top_words = [feature_names[i] for i in top_words_idx]  # Extrahiert die Top-Wörter
        print(", ".join(top_words))  # Gibt die Top-Wörter aus
    else:
        print("Keine gültigen Dokumente für dieses Cluster gefunden.")  # Gibt eine Meldung aus, wenn keine gültigen Dokumente gefunden wurden
    print()
