import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  
import skfuzzy as fuzz

# Função para gerar e-mails fictícios
def gerar_emails(num_emails):
    emails = []
    for _ in range(num_emails):
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
X_vectorizer = vetorizar.fit_transform(df['email']).toarray()

# Definindo parâmetros para Fuzzy C-Means
num_cluster = 2  # Número de clusters
m = 2  # Grau de fuzziness
error = 0.005  # Critério de parada
maxiter = 1000  # Número máximo de iterações

# Executando Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_vectorizer.T,  # Transposta dos dados vetorizados
    c=num_cluster,
    m=m,
    error=error,
    maxiter=maxiter,
    init=None
)

# Obtendo os clusters finais (maior grau de associação para cada ponto)
clusters = np.argmax(u, axis=0)

# Adicionando os clusters ao DataFrame
df['cluster'] = clusters

# Exibindo o DataFrame
print(df.head())


