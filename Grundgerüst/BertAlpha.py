import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

stopwortliste = stopwords.words('english')

reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')
reports = reports_daten['Reports'].tolist()

lemmatisierer = WordNetLemmatizer()
lemmatisierte_reports = []
for complaint in reports:
    tokens = word_tokenize(complaint.lower())
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste]
    lemmatisierte_report = ' '.join(lemmatisierte_tokens)
    lemmatisierte_reports.append(lemmatisierte_report)

model = SentenceTransformer('bert-base-nli-mean-tokens')
report_embeddings = model.encode(lemmatisierte_reports)

# Elbow-Methode zur Bestimmung der optimalen Anzahl von Clustern
range_of_clusters = range(2, 5)
sse = []

for num_clusters in range_of_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=1)
    kmeans.fit(report_embeddings)
    sse.append(kmeans.inertia_)  # Summe der quadratischen Abstände innerhalb der Cluster

# Bestimme die optimale Anzahl von Clustern
optimal_num_clusters = np.argmin(sse) + 2  # Index des minimalen SSE + 2 (da wir bei 2 Clustern anfangen)

# Führe KMeans-Clustering mit der optimalen Anzahl von Clustern durch
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=1)
kmeans.fit(report_embeddings)

# weist jeder einzelnen Beschwerde (complaint) aus der CSV-Datei eine Nummer zu die ein Thema repräsentiert
cluster_labels = kmeans.labels_

# Cluster in Text umwandeln
clustered_reports = {} # erstellt leeres Dictionary 
for cluster_idx in range(optimal_num_clusters):  # Iteriert über Clusteranzahl
    clustered_reports[cluster_idx] = []  # Erstellt für jede Clustergruppe eine leere Liste im Dictionary

# Iteriere über die Zuordnung von Beschwerden zu Clustern und füge die Beschwerden den entsprechenden Clustern im Dictionary hinzu
for idx, label in enumerate(cluster_labels): # Forschleife, die über die Indizes und (Merkmals-)Nummer in der cluster_labels Liste iteriert
    clustered_reports[label].append(reports[idx]) # Extrahiert Beschwerden mit dem spezifischen Merkmal (label) und der spezifischen Nummer (idx) aus der Liste reports und fügt sie der Liste clustered_reports hinzu

# Ausgabe der gruppierter Beschwerden für jedes Cluster:
for cluster_idx, complaints in clustered_reports.items(): # .items() gibt jedes Schlüssel-Wert-Paar im Dictionary clustered_reports aus [Cluster_idx = Schlüssel; complaint = Wert]
    print("Cluster " + str(cluster_idx) + ":")  # printet das Wort "Cluster" + "Clusternummer = cluster_idx" + ":"
    print()  # Leerzeile für bessere Lesbarkeit
    for complaint in complaints:
        print(complaint)  # Gibt jede Beschwerde im Cluster aus
        print()  # Leerzeile zw. Beschwerden
    print()  # Leerzeile zw. Clustern

def Topwords_per_cluster(clustered_reports, num_topics=1, num_words=3):
    for cluster_idx, complaints in clustered_reports.items():
        print("Cluster " + str(cluster_idx) + ":")  # printet das Wort "Cluster" + "Clusternummer = cluster_idx" + ":"

        # Überprüfe, ob die Dokumente nicht nur Stoppwörter enthalten
        valid_documents = [complaint for complaint in complaints if len(complaint.split()) > 2]  # Filtere Dokumente mit mehr als 2 Wörtern heraus
        if valid_documents:

            # LDA-Modell initialisieren und anpassen
            DTM_Vectorizer = CountVectorizer(stop_words='english')  # Initialisiere den CountVectorizer mit englischen Stoppwörtern (DTM = Dokument-Term-Matrix)
            DTM_Cluster = DTM_Vectorizer.fit_transform(valid_documents)  # Wende den CountVectorizer auf gültigen Dokumente an
            
            # Initialisiert LDA-Modell
            lda = LatentDirichletAllocation(num_topics, random_state=1) # LDA Modell versucht (bei num_topics = 1 Codezeile 62) ein Top-Thema zu finden; random_state sorgt dafür, dass Ergebnisse reproduzierbar sind (ohne random_state führt das Ausführen dieses Codes jedesmal zu anderen Ergebnissen)
            lda.fit(DTM_Cluster)  # LDA-Modell passt sich an den aktuellen Cluster an und lernt die Verteilung der Wörter/Themen

            # Ausgabe der Themen für das aktuelle Cluster
            feature_names = DTM_Vectorizer.get_feature_names_out()  
            for topic_idx, topic in enumerate(lda.components_): # enumerate gibt Index (topic_idx) und Verteilung (topic) aus / lda.components_ Attribut/Variable die die Wahrscheinlichkeitsverteilung der Wörter in den Themen des LDA-Modells darstellt
                top_words_idx = topic.argsort()[:-num_words - 1:-1]  #  Ausdruck sortiert die Indizes der Wörter nach ihrer Häufigkeit in absteigender Reihenfolge und wählt dann die ersten num_words Indizes aus
                top_words = [feature_names[i] for i in top_words_idx]  # Wählt entsprechende Wörter aus den feature_names aus
                print("Topic " + str(topic_idx) + ": " + ' '.join(top_words)) # Ausgabe der Top-Wörter für jedes Thema
        else:
            print("Keine gültigen Dokumente für dieses Cluster gefunden.")
        print()

# Ausgabe der Top-Wörter jedes Clusters
Topwords_per_cluster(clustered_reports)  # Aufruf der Funktion zum Ausgeben der Top-Wörter jedes Clusters
