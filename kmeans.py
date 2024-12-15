from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np
import pandas as pd

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

emails = gerar_emails(1000)

df = pd.DataFrame({'email': emails})

vetorizar = CountVectorizer()
x_verorized = vetorizar.fit_transform(df['email'])



num_cluster = 2

modelo = KMeans(n_clusters=num_cluster, random_state=42)

modelo.fit(x_verorized)

clusters = modelo.predict(x_verorized)

df['cluster'] = clusters

df[df['cluster']==0]

import joblib

joblib.dump(modelo, 'kmenas_model.plk')

