import pandas as pd  # # Importiert Pandas-Bibliothek und nennt sie pd
import numpy as np  # # Importiert NumPy-Bibliothek und nennt sie np
import nltk  # Importiert NLTK-Bibliothek
from nltk.corpus import stopwords  # Importiert NLTK - Stoppwortliste
from nltk.tokenize import word_tokenize  # Importiert NLTK - Tokenisierer
from nltk.stem import WordNetLemmatizer  # Importiert NLTK - Lemmatizer
from sklearn.cluster import KMeans  # # Importiert Scikit-Learn - KMeans-Modell
from sentence_transformers import SentenceTransformer  # Importiert SentenceTransformer - Modell
from sklearn.decomposition import LatentDirichletAllocation  # Importiert Scikit-Learn - LDA-Modell
from sklearn.feature_extraction.text import CountVectorizer  # Importiert Scikit-Learn - CountVectorizer


# Lädt Englische Stoppwortliste
stopwortliste = stopwords.words('english')

# Lädt CSV-Datei und füllt NaN-Werte mit leerem String auf
reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_daten['Reports'].tolist()

# Wandelt jede complaint (einzelne Beschwerde) in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports (Sammlung aller lemmatisierten Beschwerden) hinzu
lemmatisierer = WordNetLemmatizer()
lemmatisierte_reports = []
for complaint in reports:
    tokens = word_tokenize(complaint.lower())  # Tokenisierung Beschwerdetext und Umwandlung in Kleinbuchstaben
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste]  # Lemmatisierung der Tokens, entfernen von Stoppwörtern
    lemmatisierte_report = ' '.join(lemmatisierte_tokens)  # Zusammenfügen der lemmatisierten Tokens zu einem String
    lemmatisierte_reports.append(lemmatisierte_report)  # Hinzufügen des lemmatisierten Beschwerdetextes zur Liste

# BERT-Modell laden
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Lädt BERT-Modell als model

# Transformierte Texte in dichten Vektorraum
report_embeddings = model.encode(lemmatisierte_reports)  # Wendet BERT-MOdell an um Vektordarstellung für Beschwerdetexte zu erzeugen

# Führt KMeans-Clustering durch
num_clusters = 10  # Anzahl der Cluster
kmeans = KMeans(num_clusters, random_state=1)  # Initialisierung KMeans-Modell /Erstellt KMeans-Objekt (mit random_state für reprodzierbare Ergebnisse)
kmeans.fit(report_embeddings)  # Passt KMeans-Modell an Vektordarstellung (welche durch BERT erzeugt wurde) an

# weist jeder einzelnen Beschwerde (complaint) aus der CSV-Datei eine Nummer zu die ein Thema repräsentiert
cluster_labels = kmeans.labels_

# Cluster in Text umwandeln
clustered_reports = {} # erstellt leeres Dictionary 
for cluster_idx in range(num_clusters):  # Iteriert über Clusteranzahl
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


