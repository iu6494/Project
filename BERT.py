import pandas as pd # Importiert Pandas-Bibliothek und nennt sie pd
import numpy as np # Importiert NumPy-Bibliothek und nennt sie np
import nltk # Importiert NLTK-Bibliothek
from nltk.corpus import stopwords # Importiert NLTK -Stoppwortliste
from nltk.tokenize import word_tokenize # Importiert NLTK - Tokenisierer
from nltk.stem import WordNetLemmatizer # Importiert NLTK - Lemmatizer
from sklearn.cluster import KMeans # Importiert Scikit-Learn - KMeans-Modell
from sentence_transformers import SentenceTransformer # Importiert SentenceTransformer aus gleichnamiger Bibliothek
from sklearn.decomposition import LatentDirichletAllocation # Importiert Scikit-Learn - LDA-Modell
from sklearn.feature_extraction.text import TfidfVectorizer  # Importiert Scikit-Learn - TF-IDF-Vektorisierer
from sklearn.metrics import davies_bouldin_score # Importiert Scikit-Learn - Davies-Bouldin Score
from sklearn.metrics.pairwise import cosine_similarity # Importiert Scikit-Learn - Kosinus-Ähnlichkeitsmetrik

# Lädt Englische Stoppwortliste aus NLTK
stopwortliste = stopwords.words('english')

# Lädt CSV-Datei mit Pandas und ersetzt NaN-Werte durch leeren String
reports_daten = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_daten['Reports'].tolist()


# Wandelt jede Beschwerde in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports hinzu
lemmatisierer = WordNetLemmatizer() # Initialisierung WordNetLemmatizer als Variable (NLTK)
lemmatisierte_reports = [] # Erstellt leere Liste mit Variablen Namen lemmatisierte_reports
for complaint in reports: # for-Schleife die über jede einzelne Beschwerde in der Liste reports (Sammlung aller Beschwerden) iteriert
    tokens = word_tokenize(complaint.lower())  # Tokenisiert jede einzelne Beschwerden und wandelt sie in Kleinbuchstaben um (ermöglicht einheitliche Verarbeitung )
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste] # Lemmatisiert Tokens um sie auf ihre Grundform zu reduzieren (z. B. wandelt "running" in "run" um) und speichert diese Lemmas in der Variable/Liste lemmatisierte_tokens, sofern sie nicht in der Stoppwortliste enthalten sind:Lemmatisierte Tokens werden in der Liste lemmatisierte_tokens gespeichert, jedoch nur, wenn sie nicht in der Stoppwortliste enthalten sind, um irrelevante Wörter zu entfernen.
    lemmatisierte_complaint = ' '.join(lemmatisierte_tokens)  # Fügt lemmatisierte Tokens zu einem String zusammen (' '.join fügt Leerzeichen ein)
    lemmatisierte_reports.append(lemmatisierte_complaint) # Fügt lemmatisierten einzelnen Beschwerdetext zur Liste hinzu

# BERT-Modell laden
model = SentenceTransformer('bert-base-nli-mean-tokens')  # Initialisiert BERT-Modell 
report_embeddings = model.encode(lemmatisierte_reports)  # Kodiert lemmatisierte Beschwerden in Vektoren

# Initialisierung Clustering- sowie Davies-Bouldin-Parameter
best_db_index = float('inf')  # Initialisiert zu Beginn den Davies-Bouldin Index zu unendlich. Dieser Wert "unendlich" wird sukzessiv durch den nächstbesten Davies-Bouldin-Index ersetzt bis der "Beste" gefunden wurde
best_num_clusters = 7  # Initialisiert Startwert für Clusteranzahl (=> Wunschanzahl von Clustern)
min_clusters = 6  # Minimale Anzahl von Clustern
max_clusters = 8  # Maximale Anzahl von Clustern

# Iteration über verschiedene Cluster-Anzahlen
for num_clusters in range(min_clusters, max_clusters + 1): # for-Schleife die über die minimale bis zur maximalen Clusteranzahl iteriert (+1 stellt sicher, dass der max_cluster in der Iteration inkludiert wird => Range würde den Endwert [max_cluster] sonst ausschließen
    kmeans = KMeans(num_clusters, random_state=1)  # Initialisierung KMeans-Modell /Erstellt KMeans-Objekt (mit random_state für reprodzierbare Ergebnisse) aus der Scikit-learn Bibliothek
    kmeans.fit(report_embeddings) # Passt KMeans-Modell an übergebene Embedding an (s. Codezeile 34)
    cluster_labels = kmeans.labels_ # weist jeder einzelnen Beschwerde aus der CSV-Datei eine Clusterkennung zu die ein Thema repräsentiert
    db_cluster_index = davies_bouldin_score(report_embeddings, cluster_labels) # Berechnet Davies-Bouldin Index für aktuellen Cluster
    if db_cluster_index < best_db_index: # if Schleife: Vergleicht zuerst initalen best_db_index-Wert "unendlich" ('inf'; Codezeile 37) mit neu berechnetem Davies-Bouldin-Index des Clusters und ersetzt diesen bei jeder Iteration, wenn die Bedingung db_cluster_index < best_db_index erfüllt ist
        best_db_index = db_cluster_index # Aktualisiert den besten Davies-Bouldin-Index -> Überprüft, ob aktuell berechneter Davies-Bouldin-Index (db_cluster_index) besser ist als der bisher beste Index (best_db_index). Wenn ja, wird der bisher beste Index durch aktuellen Index ersetzt, um den besten Davies-Bouldin-Index zu aktualisieren.
        best_num_clusters = num_clusters # Aktualisiert die beste Clusteranzahl -> Wenn aktuell berechneter Davies-Bouldin-Index (db_cluster_index) besser ist als der bisher beste Index (best_db_index), wird die beste Clusteranzahl (best_num_clusters) aktualisiert, um die Anzahl der Cluster zu speichern, die zu diesem besseren Index führt.

# Cluster in Text umwandeln und Ausgabe der gruppierten Beschwerden für jedes Cluster
clustered_reports = {}  # Initialisierung des Dictionary für gruppierte Beschwerden
for cluster_index in range(best_num_clusters): # For-Schleife: Cluster_idx iteriert über die (zuvor berechnete beste) Clusteranzahl
    clustered_reports[cluster_index] = []  # Initialisierung der Cluster im Dictionary

# Zuordnung der Beschwerden zu den Clustern
for idx, label in enumerate(cluster_labels): # For-Schleife die über die Liste der Clusterkennungen iteriert. enumerate gibt Index und die einzelne Beschwerde zurück
    if label < best_num_clusters: # IF-Schleife: stellt sicher, dass nur die Beschwerden in den Clustern sind, die innerhalb der angegebenen Anzahl von Clustern liegen. "<" filtert so irrelevante/überflüssige Themen heraus
        clustered_reports[label].append(reports[idx])  # Wählt den Cluster aus, dem die Beschwerde zugeordnet werden soll, und fügt die einzelne Beschwerde in den entsprechenden Cluster mit der passenden Kennzahl (label) ein.

# Ausgabe der gruppierten Beschwerden für jedes Cluster
def Topwords_per_cluster(clustered_reports, num_topics=5, num_words=10):
    top_words_per_cluster = {} # Initialisiert Dictionaries für die Top-Wörter jedes Clusters

    print() # Leerzeile für bessere Lesbarkeit
    print("Zusammenfassung der Hauptthemen in den Beschwerdeclustern")
    print() # Leerzeile für bessere Lesbarkeit

    for cluster_idx, clustered_complaint in clustered_reports.items(): # FOR-Schleife, die über die gruppierten Beschwerden iteriert und sowohl den Index als auch die einzelne Beschwerde (des zugehörigen Clusters) zurückgibt; .items() wird benötigt, um Indexes und Beschwerden (das Schlüssel-Wert-Paar) extrahieren zu können.
        print("Cluster " + str(cluster_idx) + ":")  # Ausgabe "Cluster" + "Cluster-Index" + ":"
        
        # FOR-Schleife, die Beschwerden einer Liste namens cluster_specific_complaints hinzufügt, falls sie eine bestimmte Mindestwortanzahl (> als 2 Worte) überschreiten.
        cluster_specific_complaints = [complaint for complaint in clustered_complaint if len(complaint.split()) > 2]

        if cluster_specific_complaints: # IF-Schleife: ...wenn es sich um eine Beschwerde handelt die mehr als 2 Wörter hat, dann folgt folgendes...
            DTM_Vectorizer = TfidfVectorizer(stop_words='english') # es wir ein TF-IDF-Vectorizer initialisiert der englische Stoppwörter vorab herausfiltert
            DTM_Vectorizer.fit(cluster_specific_complaints) # TF-IDF-Vectorizer passt sich an die spezifischen geclusterten Beschwerden an (lernt dessen Vokabeln sowie deren Dokumenthäufigkeiten)
            DTM_Cluster = DTM_Vectorizer.transform(cluster_specific_complaints) # Tranformiert die spezifischen geclusterten Beschwerden in numerische Vektoren unter Verwendung von TF-IDF-Gewichtung
            lda = LatentDirichletAllocation(num_topics, random_state=1) # initialisiert ein Latent Dirichlet Allocation (LDA) Modell mit einer bestimmten Anzahl von Themen (num_topics); (random_state=1 um Ergebnisse reproduzierbar zu machen)
            lda.fit(DTM_Cluster) # Passt LDA-Modell an die geclusterte Dokument-Therm-Matrix an
            feature_names = DTM_Vectorizer.get_feature_names_out() # Extrahiert die Merkmalsnamen (auch als Top-Wörter bezeichnet) für jedes Cluster aus dem TF-IDF-Vektorizer, basierend auf den Wahrscheinlichkeiten in der Thema-Wort-Matrix des LDA-Modells
            top_words_per_cluster[cluster_idx] = [] # Initialisierung der Liste für die Top-Wörter in diesem Cluster
            for topic_idx, topic in enumerate(lda.components_): # For-Schleife, die über jedes Thema (topic) in der lda_components_ Matrix (s. Codezeile 112) iteriert und gleichzeitig den Index jedes Themas (topic_idx) verfolgt."
                top_words_idx = topic.argsort()[-num_words - 1:-1] # sortiert die Indizes der Wörter im Thema (topic) in absteigender Reihenfolge basierend auf ihren Wahrscheinlichkeiten und wählt dann die ersten x Indizes aus, die den höchsten Wahrscheinlichkeiten entsprechen.
                top_words = [feature_names[i] for i in top_words_idx] # Extrahiert die Top-Wörter aus feature_names (Codezeile 83) basierend auf den Indizes in top_words_idx (also den letzten 3 Top-Themen Codezeile 86)
                top_words_per_cluster[cluster_idx].append(top_words) # Fügt die Top-Wörter der Liste Top_words_per_cluster zu
                print("Topic " + str(topic_idx) + ": " + ', '.join(top_words)) # Gibt das Wort "Topic " + "Topic-Index" + ": " und die Top-Wörter aus (", " fügt zw. den einzelnen String einen Beistrich und ein Leerzeichen hinzu für bessere Lesbarkeit)
        else: # ansonsten...
            print("Keine gültigen Dokumente für dieses Cluster gefunden.")
        print() # Leerzeile für bessere Lesbarkeit

    # Initialisierung des TF-IDF-Vectorizer
    DTM_Vectorizer = TfidfVectorizer(stop_words='english') # Initialisierung des TF-IDF-Vectorizer mit englischen Stopwörtern

    # Initialisierung und Anpassung des LDA-Modells
    DTM_Vectorizer.fit(lemmatisierte_reports) # Anwendung des TF-IDF-Vectorizer
    DTM = DTM_Vectorizer.transform(lemmatisierte_reports) # TF-IDF Vectorizer passt sich an die lemmatisierte Beschwerdensammlung an (lernt dessen Vokabeln sowie deren Dokumenthäufigkeiten)
    lda = LatentDirichletAllocation(best_num_clusters, random_state=1) # Initialisiert LDA-Modell mit bester Clusteranzahl basierend auf dem zuvor ermittelten Davies-Bouldin Score; (random_state=1 um Ergebnisse reproduzierbar zu machen)
    lda.fit(DTM) # LDA-Modell lernt aus den TF-IDF-Vektoren (DTM), wie die latenten (verborgenen) Themen in den Dokumenten entdeckt und repräsentiert werden.
    feature_names = DTM_Vectorizer.get_feature_names_out() # Extrahiert die Merkmalsnamen (auch als Top-Wörter bezeichnet) für jedes Cluster aus dem TF-IDF-Vektorizer, basierend auf den Wahrscheinlichkeiten in der Thema-Wort-Matrix des LDA-Modells
    
    print("Hauptthemen für jedes Cluster im LDA-Modell mit der besten Anzahl von Clustern gruppiert:")
    print() # Leerzeile für bessere Lesbarkeit

    for topic_idx, topic in enumerate(lda.components_): # For-Schleife, die über jedes Thema (topic) in der topic_word_matrix iteriert und gleichzeitig den Index jedes Themas (topic_idx) verfolgt."
        top_words_idx = topic.argsort()[:-num_words - 1:-1] # sortiert die Indizes der Wörter im Thema (topic) in absteigender Reihenfolge basierend auf ihren Wahrscheinlichkeiten und wählt dann die ersten drei Indizes aus, die den höchsten Wahrscheinlichkeiten entsprechen.
        top_words = [feature_names[i] for i in top_words_idx] # Extrahiert die Top-Wörter aus feature_names (Codezeile 102) basierend auf den Indizes in top_words_idx (also den letzten 3 Top-Themen Codezeile 108)
        print("Topic " + str(topic_idx) + ": " + ', '.join(top_words)) # Ausgabe der Top-Wörter für jedes Thema im Cluster

    # Berechnung des c_v-Coherence-Scores für das LDA-Modell 
    def get_cv_coherence(model, corpus, texts, dictionary, coherence='c_v'): # Funktion berechnet den c_v-Coherence-Score für ein LDA-Modell und wartet auf die Paramter in Codezeile 96

    # Transformiere den Korpus in ein Array von Wortverteilungen
        top_words = model.transform(corpus)
    
    # Initialisierung von Variablen für die Cohärenzberechnung
        coherence_sum = 0.0
        num_pairs = 0

    # Iteration durch die Texte, um die Cohärenz zu berechnen
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
    
    # Überprüft, ob die Texte unterschiedlich sind, um Duplikate zu vermeiden
                if texts[i] != texts[j]: # wenn text i ungleich j ist dann...
                    num_pairs = num_pairs +1 # ...erhöhe die Anzahl der verglichenen Paare um 1

                    # Berechne Kosinus-Ähnlichkeit zw. den häufigsten Wörtern in text[i] und text[j]
                    coherence_sum = coherence_sum + cosine_similarity([top_words[i]], [top_words[j]])[0][0] # und extrahiere den Ähnlichkeitswert aus der Ähnlichkeitsmatrix und füge ihn zur Gesamtsumme hinzu
    # Berechnet Durchschnitt der Cohärenz            
        return coherence_sum / num_pairs 
    
    # Berechnung des c_v-Coherence-Scores für das LDA-Modell
    cv_coherence_score = get_cv_coherence(lda, DTM, lemmatisierte_reports, dictionary=DTM_Vectorizer, coherence='c_v') # mode1 = Art der Themenanalyse (hier LDA), Corpus = die DT Matrix, texts = die lemmatisierten Beschwerden, dictionary = der TF-IDF Vektorizer, coherence = 'c_v' = Kohärenztyp der verwendet werden soll zB u_mass, c_uci, c_npmi etc.
    
    print()
    print("Beste Anzahl von Clustern:", best_num_clusters)
    print("Davies-Bouldin Index für die beste Anzahl von Clustern:", best_db_index)
    print("c_v Coherence Score:", cv_coherence_score)
    print()

# Ausgabe der Top-Wörter für jedes Cluster
Topwords_per_cluster(clustered_reports)
