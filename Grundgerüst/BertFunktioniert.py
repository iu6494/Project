import pandas as pd  # Importiert Pandas-Bibliothek und nennt sie pd
import numpy as np  # Importiert NumPy-Bibliothek und nennt sie np
import nltk  # Importiert NLTK-Bibliothek
from nltk.corpus import stopwords  # Importiert NLTK - Stoppwortliste
from nltk.tokenize import word_tokenize  # Importiert NLTK - Tokenisierer
from nltk.stem import WordNetLemmatizer  # Importiert NLTK - Lemmatizer
from sklearn.cluster import KMeans  # Importiert Scikit-Learn - KMeans-Modell
from sentence_transformers import SentenceTransformer  # Importiert SentenceTransformer - Modell
from sklearn.decomposition import LatentDirichletAllocation  # Importiert Scikit-Learn - LDA-Modell
from sklearn.feature_extraction.text import CountVectorizer  # Importiert Scikit-Learn - CountVectorizer
from sklearn.metrics import davies_bouldin_score  # Importiert die Funktion zum Berechnen des Davies-Bouldin-Index
from sklearn.metrics.pairwise import cosine_similarity  # Importiert die Funktion zur Berechnung der Kosinus-Ähnlichkeit

# Lädt Englische Stoppwortliste
stopwortliste = stopwords.words('english')  # Laden der englischen Stopwortliste

# Lädt CSV-Datei und füllt NaN-Werte mit leerem String auf
reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')  # Laden der CSV-Datei und Auffüllen von NaN-Werten

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_daten['Reports'].tolist()  # Extrahieren der Spalte "Reports" und Umwandlung in eine Liste

# Wandelt jede Beschwerde in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports hinzu
lemmatisierer = WordNetLemmatizer()  # Initialisierung des Lemmatizers
lemmatisierte_reports = []  # Initialisierung der Liste für lemmatisierte Beschwerden
for complaint in reports:
    tokens = word_tokenize(complaint.lower())  # Tokenisierung und Umwandlung in Kleinbuchstaben
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste]  # Lemmatisierung und Entfernung von Stoppwörtern
    lemmatisierte_report = ' '.join(lemmatisierte_tokens)  # Zusammenfügen der lemmatisierten Tokens
    lemmatisierte_reports.append(lemmatisierte_report)  # Hinzufügen zur Liste lemmatisierte_reports

# BERT-Modell laden
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Laden des BERT-Modells

# Transformierte Texte in dichten Vektorraum
report_embeddings = model.encode(lemmatisierte_reports)  # Codierung der Beschwerdetexte in einen Vektorraum

# Führt KMeans-Clustering durch
num_cluster = 10  # Anzahl der Cluster
kmeans = KMeans(num_cluster, random_state=1)  # Initialisierung des KMeans-Modells
kmeans.fit(report_embeddings)  # Anpassung des Modells an die Daten
cluster_labels = kmeans.labels_  # Zuweisung der Cluster-Labels

# Berechnung des Davies-Bouldin-Index
db_index = davies_bouldin_score(report_embeddings, cluster_labels)  # Berechnung des Davies-Bouldin-Index

best_db_index = float('inf')  # Initialisierung des besten Davies-Bouldin-Index
best_num_clusters = 10  # Initialisierung der besten Anzahl von Clustern

min_clusters = 5  # Minimale Anzahl von Clustern
max_clusters = 20  # Maximale Anzahl von Clustern

# Finden der besten Anzahl von Clustern
for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=1)  # Initialisierung des KMeans-Modells
    kmeans.fit(report_embeddings)  # Anpassung des Modells an die Daten
    cluster_labels = kmeans.labels_  # Zuweisung der Cluster-Labels
    db_index = davies_bouldin_score(report_embeddings, cluster_labels)  # Berechnung des Davies-Bouldin-Index
    if db_index < best_db_index:
        best_db_index = db_index  # Aktualisierung des besten Davies-Bouldin-Index
        best_num_clusters = num_clusters  # Aktualisierung der besten Anzahl von Clustern

# Cluster in Text umwandeln
clustered_reports = {}  # Initialisierung des Dictionary für gruppierte Beschwerden
for cluster_idx in range(best_num_clusters):
    clustered_reports[cluster_idx] = []  # Initialisierung der Cluster im Dictionary

# Zuordnung der Beschwerden zu den Clustern
for idx, label in enumerate(cluster_labels):
    if label < best_num_clusters:
        clustered_reports[label].append(reports[idx])  # Hinzufügen der Beschwerden zu den entsprechenden Clustern

# Ausgabe der gruppierten Beschwerden für jedes Cluster
for cluster_idx in range(num_cluster):
    if cluster_idx in clustered_reports:
        print("Cluster " + str(cluster_idx) + ":")  # Ausgabe des Cluster-Index
        print()
        for complaint in clustered_reports[cluster_idx]:
            print(complaint)  # Ausgabe jeder Beschwerde im Cluster
        print()

print("Davies-Bouldin Index:", db_index)  # Ausgabe des Davies-Bouldin Index
print("Beste Anzahl von Clustern:", best_num_clusters)  # Ausgabe der besten Anzahl von Clustern
print("Davies-Bouldin Index für die beste Anzahl von Clustern:", best_db_index)  # Ausgabe des Davies-Bouldin Index für die beste Anzahl von Clustern
print()

# Initialisierung des CountVectorizer
DTM_Vectorizer = CountVectorizer(stop_words='english')  # Initialisierung des CountVectorizer mit englischen Stopwörtern

# Berechnung des UMass-Coherence-Scores
def umass_coherence(lda_model, dtm_cluster):
    coherence_score = 0.0  # Initialisierung des Kohärenz-Scores
    num_topics = lda_model.n_components  # Anzahl der Themen im LDA-Modell
    topic_word_matrix = lda_model.components_  # Matrix der Themen-Wörter
    vocabulary_size = dtm_cluster.shape[1]  # Größe des Vokabulars entspricht der Anzahl der Spalten in der DTM
    feature_names = list(DTM_Vectorizer.get_feature_names_out())  # Liste der Feature-Namen

    for topic_idx, topic in enumerate(topic_word_matrix):
        top_words_idx = topic.argsort()[:-3 - 1:-1]  # Auswahl der Top 3 Wörter für jedes Thema
        top_words = [feature_names[i] for i in top_words_idx]  # Auswahl der entsprechenden Wörter

        pairwise_coherence = 0.0
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                word1_idx = feature_names.index(top_words[i])  # Index des ersten Wortes
                word2_idx = feature_names.index(top_words[j])  # Index des zweiten Wortes
                word_pair_similarity = cosine_similarity(topic[word1_idx].reshape(1, -1), topic[word2_idx].reshape(1, -1))  # Berechnung der Ähnlichkeit
                pairwise_coherence += word_pair_similarity  # Aggregation der Ähnlichkeiten
        
        coherence_score += pairwise_coherence / (len(top_words) * (len(top_words) - 1) / 2)  # Berechnung der Kohärenz für das Thema
        
    return coherence_score / num_topics  # Durchschnittliche Kohärenz über alle Themen

# Initialisierung und Anpassung des LDA-Modells
DTM_Cluster = DTM_Vectorizer.fit_transform(lemmatisierte_reports)  # Anwendung des CountVectorizer
lda = LatentDirichletAllocation(best_num_clusters, random_state=1)  # Initialisierung des LDA-Modells
lda.fit(DTM_Cluster)  # Anpassung des LDA-Modells

# Berechnung des UMass-Coherence-Scores
umass_score = umass_coherence(lda, DTM_Cluster)  # Berechnung des UMass-Coherence-Scores
print("UMass Coherence Score:", umass_score)  # Ausgabe des UMass-Coherence-Scores
print()

# Ausgabe der Top-Wörter für jedes Cluster
def Topwords_per_cluster(clustered_reports, num_topics=1, num_words=3):
    for cluster_idx, complaints in clustered_reports.items():
        print("Cluster " + str(cluster_idx) + ":")  # Ausgabe des Cluster-Index
        print()

        # Überprüfung, ob die Dokumente nicht nur Stoppwörter enthalten
        valid_documents = [complaint for complaint in complaints if len(complaint.split()) > 2]
        if valid_documents:

            # Initialisierung und Anpassung des LDA-Modells
            DTM_Vectorizer = CountVectorizer(stop_words='english')  # Initialisierung des CountVectorizer mit englischen Stopwörtern
            DTM_Cluster = DTM_Vectorizer.fit_transform(valid_documents)  # Anwendung des CountVectorizer
            lda = LatentDirichletAllocation(num_topics, random_state=1)  # Initialisierung des LDA-Modells
            lda.fit(DTM_Cluster)  # Anpassung des LDA-Modells

            # Ausgabe der Top-Wörter für jedes Thema im Cluster
            feature_names = DTM_Vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-num_words - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                print("Topic " + str(topic_idx) + ": " + ' '.join(top_words))  # Ausgabe der Top-Wörter für jedes Thema
        else:
            print("Keine gültigen Dokumente für dieses Cluster gefunden.")  # Ausgabe bei fehlenden Dokumenten
        print()

# Ausgabe der Top-Wörter für jedes Cluster
Topwords_per_cluster(clustered_reports)  # Aufruf der Funktion zur Ausgabe der Top-Wörter jedes Clusters
