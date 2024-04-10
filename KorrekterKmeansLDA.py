import pandas as pd  # Importiert Pandas-Bibliothek und nennt sie pd
import numpy as np  # Importiert NumPy-Bibliothek und nennt sie np
import nltk  # Importiert NLTK-Bibliothek
from nltk.corpus import stopwords  # Importiert NLTK -Stoppwortliste
from nltk.tokenize import word_tokenize  # Importiert NLTK - Tokenisierer
from nltk.stem import WordNetLemmatizer  # Importiert NLTK - Lemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer  # Importiert Scikit-Learn - TF-IDF-Vektorisierer
from sklearn.cluster import KMeans  # Importiert Scikit-Learn - KMeans-Modell
from sklearn.decomposition import LatentDirichletAllocation  # Importiert Scikit-Learn - LDA-Modell

# Lädt Englische Stoppwortliste
stoppwortliste = stopwords.words('english')

# Lädt CSV-Datei und füllt NaN-Werte mit leerem String auf
reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_daten['Reports'].tolist()

# Wandelt jede complaint (einzelne Beschwerde) in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports (Sammlung aller lemmatisierten Beschwerden) hinzu
lemmatisierer = WordNetLemmatizer()
lemmatisierte_reports = []
for complaint in reports:
    tokens = word_tokenize(complaint.lower())  # Tokenisierung Beschwerdetext und Umwandlung in Kleinbuchstaben
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stoppwortliste]  # Lemmatisierung der Tokens, entfernen von Stoppwörtern
    lemmatisierte_complaint = ' '.join(lemmatisierte_tokens)  # Zusammenfügen der lemmatisierten Tokens zu einem String
    lemmatisierte_reports.append(lemmatisierte_complaint)  # Hinzufügen des lemmatisierten Beschwerdetextes zur Liste

# TF-IDF erstellen
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))  # Initialisierung des TF-IDF-Vektorisierers, mit engl. Stoppwortliste und Trigrammen
tfidf_matrix = tfidf_vectorizer.fit_transform(lemmatisierte_reports)  # Anwendung des TF-IDF-Vektorisierers auf lemmatisierte Beschwerdetexte

# Führt KMeans-Clustering durch
num_clusters = 10  # Anzahl der Cluster
kmeans = KMeans(num_clusters, random_state=1)  # Initialisierung KMeans-Modell /Erstellt KMeans-Objekt (mit random_state für reprodzierbare Ergebnisse)
kmeans.fit(tfidf_matrix)  # Passt KMeans-Modell an TF-IDF-Matrix an

# weist jeder einzelnen Beschwerde aus der CSV-Datei eine Nummer zu die ein Thema repräsentiert
cluster_labels = kmeans.labels_

# Die Zentren der Cluster erhalten
cluster_centers = kmeans.cluster_centers_

# Die Beschwerden in jedem Cluster ausgeben
for cluster_idx in range(num_clusters): # Forschleife, die jede Clusternummer(idx) iteriert
    print("Cluster " + str(cluster_idx)+":") # printet das Wort "Cluster" + "Clusternummer = cluster_idx" + ":"
    print() # Leerzeile für bessere Lesbarkeit
    
    # Indizes der Beschwerden im Cluster erhalten
    # => Hier erhält man ein Tupel von Arrays, wobei jedes Array die Indizes der Beschwerden enthält, die dem entsprechenden Cluster entsprechen.
    # => Die [0] am Ende ermöglicht, dass nur diejenigen Beschwerden ausgegeben werden die dem aktuellen Cluster entsprechen
    cluster_indices = np.where(cluster_labels == cluster_idx)[0]
    
    for idx in cluster_indices:
        print(reports[idx]) # printet jede einzelne Beschwerde in reports, welche die Bedingung cluster_labels == cluster_idx erfüllen
        print()
    print() # gibt leere Zeile aus, um in der Ausgabe eine visuelle Trennung zw. den einzelnen Clustern zu erzeugen.
    print()




# Anzahl der Themen für LDA
num_topics = 1 # Anzahl der Top-Themen die pro Cluster ausgegeben werden sollten

# Initialisieren einer leeren Liste um für jedes Cluster ein seperates LDA-Modell zu erstellen und speichern
lda_models = []

# Iteration über jeden Cluster für LDA-Modellbildung
for cluster_idx in range(num_clusters):
    # Extrahieren der Beschwerden für das aktuelle Cluster
    cluster_indices = np.where(cluster_labels == cluster_idx)[0]
    cluster_complaints = [lemmatisierte_reports[i] for i in cluster_indices] # Liste enthält alle lemmatisierten Beschwerden, die dem aktuellen Cluster entsprechen, basierend auf den cluster_indices

    # Erstellt TF-IDF-Matrix für aktuelles Cluster [transform, statt fit_transform, da ansonsten die IDF-Statistik (in Codezeile 32) neu berechnet werden würde => führt zu Inkonsistenzen und Falschinterpretationen]
    tfidf_matrix_cluster = tfidf_vectorizer.transform(cluster_complaints) # aufgrund Forschleife wird jeder Clusterindex iteriert und diese Codezeile erstellt für jedes Cluster eine eigene TF-IDF Matrix 

    # Initialisieren und Anpassen des LDA-Modells für das aktuelle Cluster
    lda = LatentDirichletAllocation(num_topics=1, random_state=1) # LDA Modell versucht (bei num_topics = 1) ein Top-Thema zu finden; random_state sorgt dafür, dass Ergebnisse reproduzierbar sind (ohne random_state führt das Ausführen dieses Codes jedesmal zu anderen Ergebnissen )
    lda.fit(tfidf_matrix_cluster) # LDA-Modell passt sich an den aktuellen Cluster an und lernt die Verteilung der Wörter/Themen

    # Fügt fertiges LDA-Modell der Liste lda_Modell (Codezeile 65) hinzu
    lda_models.append(lda)




    # Ausgabe der Themen für das aktuelle Cluster
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_): # enumerate gibt Index (topic_idx) und Verteilung (topic) aus / lda.components_ Attribut/Variable die die Wahrscheinlichkeitsverteilung der Wörter in den Themen des LDA-Modells darstellt
        print("Cluster " + str(cluster_idx) + ", Topic " + str(topic_idx) + ":") # printet das Wort "Cluster" + "Clusternummer = cluster_idx" + ":" + Topic " + str(topic_idx) + ":"
        print()
        print(", ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))  # gibt die 5 wichtigsten Wörter aus die jedes Top-Themen-Cluster beschreiben
        print()
        print()


        