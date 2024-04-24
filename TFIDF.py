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

# Lädt Englische Stoppwortliste aus NLTK
stopwortliste = stopwords.words('english')

# Lädt CSV-Datei mit Pandas und ersetzt NaN-Werte durch leeren String: Diese Zeile liest die CSV-Datei "studentreports.csv" mit der Pandas-Bibliothek ein und ersetzt dabei eventuelle NaN (Not a Number)-Werte mit .fillna durch leere Strings.
reports_spalte = pd.read_csv('/home/xstarcroftx/Aufgabe 1/studentreports.csv').fillna('')

# Extrahiert Spalte "Reports" aus CSV-Datei "studentreports.csv" und wandelt die Spalte in Liste um
reports = reports_spalte['Reports'].tolist()

# Wandelt jede complaint (einzelne Beschwerde) in Kleinbuchstaben um, lemmatisiert sie und fügt sie der Liste lemmatisierte_reports (Sammlung aller lemmatisierten Beschwerden) hinzu (NLTK)
lemmatisierer = WordNetLemmatizer() # Initialisiert WordNetLemmatizer als Variable (NLTK)
lemmatisierte_reports = [] # Erstellt leere Liste mit Variablen Namen lemmatisierte_reports
for complaint in reports: # for-Schleife die über jede einzelne Beschwerde in der Liste reports (Sammlung aller Beschwerden) iteriert
    tokens = word_tokenize(complaint.lower()) # Tokenisiert jede einzelne Beschwerden und wandelt sie in Kleinbuchstaben um (ermöglicht einheitliche Verarbeitung )
    lemmatisierte_tokens = [lemmatisierer.lemmatize(token) for token in tokens if token not in stopwortliste and token.isalnum()] # Lemmatisiert Tokens um sie auf ihre Grundform zu reduzieren (z. B. wandelt "running" in "run" um) und speichert diese Lemmas in der Variable/Liste lemmatisierte_tokens, sofern sie nicht in der Stoppwortliste enthalten sind:Lemmatisierte Tokens werden in der Liste lemmatisierte_tokens gespeichert, jedoch nur, wenn sie nicht in der Stoppwortliste enthalten sind, um irrelevante Wörter zu entfernen. token.isalnum() wird verwendet, um sicherzustellen, dass nur alphanumerische Zeichen in die Liste der lemmatisierten Tokens aufgenommen werden
    lemmatisierte_complaint = ' '.join(lemmatisierte_tokens) # Fügt lemmatisierte Tokens zu einem String zusammen (' '.join fügt Leerzeichen ein)
    lemmatisierte_reports.append(lemmatisierte_complaint) # Fügt lemmatisierten einzelnen Beschwerdetext zur Liste hinzu

# TF-IDF erstellen
tfidf_vectorizer = TfidfVectorizer(stop_words='english') # Initialisiert TF-IDF-Vektorisierer, mit engl. Stoppwortliste
tfidf_vectorizer.fit(lemmatisierte_reports)  # TF-IDF Vectorizer passt sich an die lemmatisierte Beschwerdensammlung an (lernt dessen Vokabeln sowie deren Dokumenthäufigkeiten)
tfidf_matrix = tfidf_vectorizer.transform(lemmatisierte_reports) # Tranformiert lemmatisierte Beschwerdensammlung in numerische Vektoren unter Verwendung von TF-IDF-Gewichtung

# Initialisierung Clustering- sowie Davies-Bouldin-Parameter
best_db_index = float('inf')  # Initialisiert zu Beginn den Davies-Bouldin Index zu unendlich. Dieser Wert "unendlich" wird sukzessiv durch den nächstbesten Davies-Bouldin-Index ersetzt bis der "Beste" gefunden wurde
best_num_clusters = 5  # Initialisiert Startwert für Clusteranzahl (=> Wunschanzahl von Clustern)
min_clusters = 5  # Minimale Anzahl von Clustern
max_clusters = 15  # Maximale Anzahl von Clustern


# Iteration über verschiedene Cluster-Anzahlen
for num_clusters in range(min_clusters, max_clusters + 1): # for-Schleife die über die minimale bis zur maximalen Clusteranzahl iteriert (+1 stellt sicher, dass der max_cluster in der Iteration inkludiert wird => Range würde den Endwert [max_cluster] sonst ausschließen
    kmeans = KMeans(num_clusters, random_state=1)  # Initialisierung KMeans-Modell /Erstellt KMeans-Objekt (mit random_state für reprodzierbare Ergebnisse) aus der Scikit-learn Bibliothek
    kmeans.fit(tfidf_matrix) # Passt KMeans-Modell an die übergebene TF-IDF-Matrix an (s. Codezeile 33)
    cluster_labels = kmeans.labels_ # weist jeder einzelnen Beschwerde aus der CSV-Datei eine Clusterkennung zu die ein Thema repräsentiert
    db_cluster_index = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels) # Berechnet Davies-Bouldin Index für aktuellen Cluster
    if db_cluster_index < best_db_index:  # if Schleife: Vergleicht zuerst initalen best_db_index-Wert "unendlich" ('inf'; Codezeile 37) mit neu berechnetem Davies-Bouldin-Index des Clusters und ersetzt diesen bei jeder Iteration, wenn die Bedingung db_cluster_index < best_db_index erfüllt ist
        best_db_index = db_cluster_index  # Aktualisiert den besten Davies-Bouldin-Index -> Überprüft, ob aktuell berechneter Davies-Bouldin-Index (db_cluster_index) besser ist als der bisher beste Index (best_db_index). Wenn ja, wird der bisher beste Index durch aktuellen Index ersetzt, um den besten Davies-Bouldin-Index zu aktualisieren.
        best_num_clusters = num_clusters  # Aktualisiert die beste Clusteranzahl -> Wenn aktuell berechneter Davies-Bouldin-Index (db_cluster_index) besser ist als der bisher beste Index (best_db_index), wird die beste Clusteranzahl (best_num_clusters) aktualisiert, um die Anzahl der Cluster zu speichern, die zu diesem besseren Index führt.

# Ausgabe aller Beschwerden für jedes der optimalen Cluster basierend auf dem besten Davies-Bouldin-Index

for cluster_index in range(best_num_clusters): # For-Schleife: Cluster_idx iteriert über die (zuvor berechnete beste) Clusteranzahl
    print() # Leerzeile für bessere Lesbarkeit
    print("Cluster " + str(cluster_index) + ":") # Gibt "Cluster" +"Clusternummer(idx)" + ":" aus"
    for idx, complaint in enumerate(reports): # Initiiert For-Schleife und iteriert mittels idx & complaint über die reports Liste. idx repräsentiert den Index der aktuellen Beschwerde in der Liste reports, und complaint repräsentiert den Inhalt dieser Beschwerde
        if cluster_labels[idx] == cluster_index:  # wenn die Clusterkennung (label) der Beschwerde an der Index Stelle (über die gerade iteriert wird) mit der Clusternummer übereinstimmt... 
            print(complaint)  # ...dann soll die Beschwerde (an der Position der Index Stelle) ausgegeben werden

print()
print("Beste Anzahl von Clustern:", best_num_clusters)  # Gibt beste Clusteranzahl aus
print("Davies-Bouldin Index für die beste Anzahl von Clustern:", best_db_index)  # Gibt besten Davies-Bouldin Index aus

# Initialisierung des LDA-Modells
num_topics = 1  # Anpassung der Anzahl der Themen
num_words = 3   # Anpassung der Anzahl der Wörter pro Thema

# Initialisierung und Anpassung des LDA-Modells
lda = LatentDirichletAllocation(best_num_clusters, random_state=1) # Initialisiert LDA-Modell mit bester Clusteranzahl basierend auf dem zuvor ermittelten Davies-Bouldin Score (random_state=1 um Ergebnisse reproduzierbar zu machen)
lda.fit(tfidf_matrix) # LDA-Modell lernt aus den TF-IDF-Vektoren (tfidf_matrix), wie die latenten (verborgenen) Themen in den Dokumenten entdeckt und repräsentiert werden.

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
cv_coherence_score = get_cv_coherence(lda, tfidf_matrix, lemmatisierte_reports, dictionary=tfidf_vectorizer, coherence='c_v') # mode1 = Art der Themenanalyse (hier LDA), Corpus = die TF-IDF Matrix, texts = die lemmatisierten Beschwerden, dictionary = der TF-IDF Vektorizer, coherence = 'c_v' = Kohärenztyp der verwendet werden soll zB u_mass, c_uci, c_npmi etc.
print("c_v Coherence Score:", cv_coherence_score) # gibt c_v Coherence Score: xy aus
print() # Leerzeile für bessere Lesbarkeit

# Initialisierung des LDA-Modells außerhalb der Schleife
lda = LatentDirichletAllocation(num_topics, random_state=1) # initialisiert ein Latent Dirichlet Allocation (LDA) Modell mit einer bestimmten Anzahl von Themen (num_topics); (random_state=1 um Ergebnisse reproduzierbar zu machen)

# Iteration über die beste Clusteranzahl basierend auf dem Davies-Bouldin-Index
print ('Topwörter der Geclusterten Beschwerden:')
print()
for cluster_index in range(best_num_clusters): # For-Schleife: Cluster_idx iteriert über die (zuvor berechnete beste) Clusteranzahl
    print("Cluster " + str(cluster_index) + ":") # Gibt "Cluster" +"Clusternummer(idx)" + ":" aus"
    cluster_specific_complaints = [lemmatisierte_reports[idx] for idx in range(len(reports)) if cluster_labels[idx] == cluster_index] # Extrahiert gültige Beschwerden für jedes Cluster, wobei die Clusterkennung (label) mit der Clusternummer (index) übereinstimmen muss. Wenn die Clusterkennung und die Clusternummer nicht übereinstimmen, wird die lemmatisierte Beschwerde nicht in die Liste der gültigen Dokumente aufgenommen.
    if cluster_specific_complaints: # Überprüft, ob für das aktuelle Cluster gültige Beschwerden vorhanden sind (also Beschwerden bei denen die Clusterkennung (Label) mit der Clusternummer (Index) übereinstimmt
        DTM_Cluster = tfidf_matrix[cluster_labels == cluster_index]
        lda.fit(DTM_Cluster) # Passt das Latent Dirichlet Allocation (LDA)-Modell an die Dokument-Therm-Matrix (DTM) des spezifischen Clusters an, um die latenten (verborgenen) Themen in diesem Cluster zu identifizieren und zu repräsentieren.
        feature_names = tfidf_vectorizer.get_feature_names_out() # Extrahiert die Merkmalsnamen (auch als Top-Wörter bezeichnet) für jedes Cluster aus dem TF-IDF-Vektorizer, basierend auf den Wahrscheinlichkeiten in der Thema-Wort-Matrix des LDA-Modells
        for topic_idx, topic in enumerate(lda.components_): # For-Schleife, die über jedes Thema (topic) in der n_components matrix iteriert und gleichzeitig den Index jedes Themas (topic_idx) verfolgt."
            top_words_idx = topic.argsort()[:-num_words - 1:-1] # sortiert die Indizes der Wörter im Thema (topic) in absteigender Reihenfolge basierend auf ihren Wahrscheinlichkeiten und wählt dann die ersten drei Indizes aus, die den höchsten Wahrscheinlichkeiten entsprechen.
            top_words = [feature_names[i] for i in top_words_idx] # Extrahiert die Top-Wörter aus feature_names (Codezeile 100) basierend auf den Indizes in top_words_idx (also den letzten 3 Top-Themen Codezeile 102)
            print(", ".join(top_words)) # Gibt die Top-Wörter aus (", " fügt zw. den einzelnen String einen Beistrich und ein Leerzeichen hinzu für bessere Lesbarkeit)
    else:
        print("Keine gültigen Dokumente für dieses Cluster gefunden.")
    print() # Leerzeile für bessere Lesbarkeit