from sklearn.feature_extraction.text import CountVectorizer  
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Função para gerar e-mails fictícios
def gerar_emails(num_emails):
    emails = []
    for _ in range(num_emails):
        # Gerando textos aleatórios
        if np.random.rand() < 0.5:  # 50% de chance de ser SPAM
            emails.append("Compre agora a oferta de " + np.random.choice(["grátis", "desconto", "promoção", "prêmio"]))
        else:  # 50% de chance de ser NÃO SPAM
            emails.append("Informação sobre " + np.random.choice(["reunião", "projeto", "evento", "atualização"]) + " para você.")
    return emails

# Gerando os e-mails fictícios
emails = gerar_emails(1000)
df = pd.DataFrame({'email': emails})

# Vetorizando os textos
vetorizar = CountVectorizer()
x_vectorized = vetorizar.fit_transform(df['email']).toarray()

# Calculando a matriz de distâncias
distance_matrix = calculate_distance_matrix(x_vectorized)

# Definindo os medoids iniciais
initial_medoids = [0, 1]

# Criando o modelo de K-Medoids
kmedoid_instance = kmedoids(distance_matrix, initial_medoids)

# Executando o algoritmo
kmedoid_instance.process()
clusters = kmedoid_instance.get_clusters()

# Atribuindo rótulos aos dados
df['cluster'] = -1  # Inicializando com -1 (sem cluster)
for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        df.at[index, 'cluster'] = cluster_id

# Exibindo o DataFrame com os clusters
print("\nDataFrame com clusters:")
print(df.head())

# Visualizando os resultados
# Para visualização, usaremos apenas as duas primeiras dimensões dos dados vetorizados
plt.figure(figsize=(10, 6))
for cluster_id in range(len(clusters)):
    indices = clusters[cluster_id]
    plt.scatter(x_vectorized[indices, 0], x_vectorized[indices, 1], label=f'Cluster {cluster_id}')
plt.title('K-Medoids Clustering dos E-mails')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()





