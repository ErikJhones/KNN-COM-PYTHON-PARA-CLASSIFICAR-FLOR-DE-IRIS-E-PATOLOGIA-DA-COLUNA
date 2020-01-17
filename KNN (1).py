
# coding: utf-8

# In[ ]:


#https://juliocprocha.blog/2016/05/23/k-nearest-neighbors-seu-primeiro-algoritmo-de-machine-learning/
# código do método de classificação KNN (K-nearest neighbors)


# In[1]:


#bibliotecas usadas
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import math

import matplotlib

import matplotlib.pyplot as plt

import random

from sklearn.preprocessing import MaxAbsScaler

import math

import numpy as np

from scipy import stats


# In[2]:


class Rotulo_com_distancia(object): 
    def __init__(self, padrao, distancia): 
        self.__padrao = padrao 
        self.__distancia = distancia 

    def __repr__(self): 
        return "Rotulo:%s Distancia:%s" % (self.__padrao, self.__distancia)

    def get_padrao(self): 
        return self.__padrao

    def get_distancia(self): 
        return self.__distancia


# In[3]:


#plotar grafico scarterplot
def plotar_com_reta(padroes, saidas, p1, p2):
    
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    
    n_p = [] #vetor de pontos de coloracao
    cord = 0.001 #variavel da dimensao do ponto
    #cira um vetor de posicoes bem pequenas para ser plotadas no grafico para colorir
    for j in range(100):
        nov = []
        cord2 = 0.001

        for i in range(100):
            nov = []
            nov.append(cord)
            nov.append(cord2)
        
            n_p.append(nov)
       
            cord2+=0.01
        
        cord+=0.01
    
    reta = [p1, p2]
    
    #colorir o grafico de acordo com  a reta
    for y in n_p:
        x = y[0]
        z = y[1]
        equacao_reta = ((reta[0][1] - reta[1][1]) * x) + ((reta[1][0] - reta[0][0]) * z) + ((reta[0][0] * reta[1][1]) - (reta[1][0] * reta[0][1]))

        if ( equacao_reta < 0):
            plt.scatter(y[0], y[1], s=40, c="green", alpha=0.2)
        elif (equacao_reta > 0): 
              
            plt.scatter(y[0], y[1], s=40, c="blue", alpha=0.2)
    
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="black", alpha=0.9)
        else:
            #print("lepra")  
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot(p1, p2) #reta
    plt.show()

#plotar grafico scarterplot
def plotar_sem_reta(padroes):
    
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    '''
    n_p = [] #vetor de pontos de coloracao
    cord = 0.001 #variavel da dimensao do ponto
    #cira um vetor de posicoes bem pequenas para ser plotadas no grafico para colorir
    for j in range(100):
        nov = []
        cord2 = 0.001

        for i in range(100):
            nov = []
            nov.append(cord)
            nov.append(cord2)
        
            n_p.append(nov)
       
            cord2+=0.01
        
        cord+=0.01
    
    reta = [p1, p2]
    
    #colorir o grafico de acordo com  a reta
    for y in n_p:
        x = y[0]
        z = y[1]
        equacao_reta = ((reta[0][1] - reta[1][1]) * x) + ((reta[1][0] - reta[0][0]) * z) + ((reta[0][0] * reta[1][1]) - (reta[1][0] * reta[0][1]))

        if ( equacao_reta < 0):
            plt.scatter(y[0], y[1], s=40, c="green", alpha=0.2)
        elif (equacao_reta > 0): 
              
            plt.scatter(y[0], y[1], s=40, c="blue", alpha=0.2)
    '''
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="blue", alpha=0.9)
        elif (y[tam]==0.5):
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
        else:
            plt.scatter(y[2], y[3], s=20, c="green", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot(p1, p2) #reta
    plt.show()
    
#mplota o grafico media dos acertos vs epocas
def plot_media_acerto(epocas, valores):
    
    matplotlib.pyplot.plot(epocas, valores)
    matplotlib.pyplot.title('Taxa média de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Percentual Media')
    matplotlib.pyplot.ylim(50, 100)

    matplotlib.pyplot.show()

#plota o grafico variancia dos acertos vs epocas
def plot_vari_acerto(epocas, vari):
    
    matplotlib.pyplot.plot(epocas, vari)                       
    matplotlib.pyplot.title('Variancia de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Variancia')
    matplotlib.pyplot.ylim(0, 60)

    matplotlib.pyplot.show()
    
#plota o grafico do desvio padrao vs epoca
def plot_desv_acerto(epocas, desv):
    
    matplotlib.pyplot.plot(epocas, desv)                       
    matplotlib.pyplot.title('Desvio padrao de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Desvio padrao')
    matplotlib.pyplot.ylim(0, 10)

    matplotlib.pyplot.show()

    
#funcao de embaralhamento dos padroes
def embaralhar(padrao):
    
    padroes = padrao.copy()
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    saida_teste = []
    saida_treino = []

    random.shuffle(padroes) #embaralha o veto padroes
        
    l = len(padroes)

    lista = list(range(l)) #faz uma lista de 0 até tamanho do vetor padroes
    
    random.shuffle(lista) #embaralha o vetor lista
    
    tam_teste = int(0.2 * l) #tcalculo quanto é 20% da quantidae de padroes
        
    y = 0 #variavel auxiliar
        
    teste_tam = 0
    treino_tam = 0
    
    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
            teste_tam+=1
            teste.append(padroes[x].copy()) #passa um vator de padrao para o vetor de teste
                
        else:
            treino_tam+=1
            treino.append(padroes[x].copy())
        y+=1
    
    #povoa os vetores de saida
    for i in range(teste_tam):
        saida_teste.append(teste[i].pop(len(teste[i])-1))
    
    #povoa o vetor de treino
    for j in range(treino_tam):
        saida_treino.append(treino[j].pop(len(treino[j])-1))
    
    return treino, teste, saida_treino, saida_teste
    

#mplota o grafico media dos acertos vs epocas
def plot_media_acerto(epocas, valores):
    
    matplotlib.pyplot.plot(epocas, valores)
    matplotlib.pyplot.title('Taxa média de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Percentual Media')
    matplotlib.pyplot.ylim(0, 100)

    matplotlib.pyplot.show()

#plota o grafico variancia dos acertos vs epocas
def plot_vari_acerto(epocas, vari):
    
    matplotlib.pyplot.plot(epocas, vari)                       
    matplotlib.pyplot.title('Variancia de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Variancia')
    matplotlib.pyplot.ylim(0, 60)

    matplotlib.pyplot.show()
    
#plota o grafico do desvio padrao vs epoca
def plot_desv_acerto(epocas, desv):
    
    matplotlib.pyplot.plot(epocas, desv)                       
    matplotlib.pyplot.title('Desvio padrao de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Desvio padrao')
    matplotlib.pyplot.ylim(0, 10)

    matplotlib.pyplot.show()  
    
def plotar_grafico_distancia(padroes, k_padroes, teste, vetor_teste):
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    
    
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="blue", alpha=0.9)
        elif (y[tam]==0.5):
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
        else:
            plt.scatter(y[2], y[3], s=20, c="green", alpha=0.9)
    
    for j in vetor_teste:
       
        plt.scatter(j[2], j[3], s=20, c="black", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
    
    for i in k_padroes:
        
        plt.scatter(teste[2], teste[3], s=30, c="black", alpha=0.9)
        p1 = [i[2], i[3]]
        p2 = [teste[2], teste[3]]
       
        
       
        plt.plot([i[2], teste[2]], [i[3], teste[3]])
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot([1,1], [0.5, 0.5]) #reta
    plt.show()
    
def taxa_media_acerto_teste(taxa_acerto):
    media = int(np.mean(taxa_acerto)) #media dos valore da primeira linha da imagem
    return media

def variancia_acerto_teste(taxa_acerto):
    variancia = int(np.var(taxa_acerto)) #variancia da primeia linha da imagem   
    return variancia

def desvio_padrao_teste(variancia):
    desvio = int(math.sqrt(variancia)) #desvio padrao
    return desvio       


# In[19]:


#métodos

def moda(k_padroes):
    padroes = [] #GUARDA os vetores de treino selecionados
    valores = []
    
    for i in k_padroes:
        padroes.append(i.get_padrao())
    
   
    for i in padroes:
        valores.append(i[len(i)-1])
        
 
    #moda
    mode = stats.mode(valores)                 
    return mode[0]

def padroes_proximos(classes):
    v = []
    for i in classes:
        v.append(i.get_padrao())
    return v

#verifica se houve acerto ou não
def verifica_acerto(rotulo, classe):
    #print("\n\n\n*****classes******\n", classe, rotulo,"\n\n\n")
    if (rotulo[len(rotulo) - 1] == classe):
        return 1
    else:
        return 0

def calcular_distancia(x, y):
    tam_x = len(x)
    tam_y = len(y)
    hyp = 0
    
    for i in range(tam_x):
        hyp = (x[i] - y[i])**2 + hyp
    hyp = math.sqrt(hyp)
    return hyp

def verificar_classe_padrao(padrao, k): # K é o número de classes a separar os padroes
    
    epocas = []
    classes = [] #vetor das k distancias mais proximas do padrao.
    classe_epoca = [] #gurada as modas das epocas
    teste = []
    treino = []
    s_treino = []
    s_teste = []
    acerto_epoca = [] #acertos em porcento por epoca
    
    epoca = 1
    
    while(epoca <= 30):
        
        acertos = 0
        treino, teste, s_treino, s_teste  = embaralhar(padrao.copy())#
        
        #para cada rótulo no vetor de teste
        for i, si in zip(teste, s_teste):
            classe = 0
            
            #para cada rótulo no vetor de treino
            distancia_ordenada = []
            distancias = [] #lista de distancias encontradas entre i e j
            
            for j, sj in zip(treino,s_treino):
               
                distancia = calcular_distancia(i, j) #calcula distancia euclidiana entre i e j
                
                j.append(sj)
                distancias.append(Rotulo_com_distancia(j.copy(), distancia)) #guarda a distancia no vetor distancias
                j.pop(len(j)-1)
                
            distancia_ordenada = sorted(distancias, key = Rotulo_com_distancia.get_distancia) #ordena o vetor de distancias em forma decrescente 
            classes = distancia_ordenada[0:k] #classes mais proximas
            
            #plotar grafico com reta para os padroes achados mais proximos
            #plotar_grafico_distancia(padrao, padroes_proximos(classes), i, teste)
           
            i.append(si)
            acertos+=verifica_acerto(i, moda(classes)) #verifica se teste i pertence a classe obtida e em caso sim, incrementa contador
            i.pop(len(i)-1)
            
        #print("\n\nacertos\n\n", acertos)
        acerto_epoca.append(int((acertos * 100) / len(teste))) #moda
        #print("*****acertos epocas******\n", acerto_epoca)
        epocas.append(epoca)
        epoca+=1
    
    #print("*************************************************************************************************************")
    
    media = taxa_media_acerto_teste(acerto_epoca)
    vari = variancia_acerto_teste(acerto_epoca)
    desv = desvio_padrao_teste(vari)
    return acerto_epoca, media, vari, desv, epocas

        


# In[20]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo, classe):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
        #verifica quantas classes quero separar os padroes    
        if (classe == 2):    
            if(cont <50):
                x[len(x)-1] = 1
            else:
                x[len(x)-1] = 0
        elif (classe == 3):
            if(cont < 50):
                x[len(x)-1] = 0
            elif (cont >= 50 and cont < 100):
                x[len(x)-1] = 0.5
            else:
                x[len(x)-1] = 1
        
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close()    
    return padroes

c = ler_arquivo("iris.data", 2)
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)
f = dados.tolist() #vetor de padroes da iris 


#w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(f.copy(), saidas)

#print("****padroes*****\n", f, len(f))
#p1, p2 = coordenadas(w)
plotar_sem_reta(dados)

valores_acertos = []
#escolha a quantidade de classes a separar os padroes
valores_acertos, media, vari, desv, epocas = verificar_classe_padrao(f, 2) 

print("****valores dos acertos*****","\n", valores_acertos, "\nmedia", media, "\nvariancia ", vari,"\n desvio padrao", desv, "\nepocas", epocas)
plot_media_acerto(epocas, valores_acertos)


# In[21]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo, classe):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
        #verifica quantas classes quero separar os padroes    
        if (classe == 2):    
            if(cont <50):
                x[len(x)-1] = 1
            else:
                x[len(x)-1] = 0
        elif (classe == 3):
            if(cont < 50):
                x[len(x)-1] = 0
            elif (cont >= 50 and cont < 100):
                x[len(x)-1] = 0.5
            else:
                x[len(x)-1] = 1
        
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close()    
    return padroes

c = ler_arquivo("iris.data", 3)
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)
f = dados.tolist() #vetor de padroes da iris 


#w, taxa_media, taxa_vari, taxa_desv, epoca = treinamento(f.copy(), saidas)

#print("****padroes*****\n", f, len(f))
#p1, p2 = coordenadas(w)
plotar_sem_reta(dados)

valores_acertos = []
#escolha a quantidade de classes a separar os padroes
valores_acertos, media, vari, desv, epocas = verificar_classe_padrao(f, 3) 

print("****valores dos acertos*****","\n", valores_acertos, "\nmedia", media, "\nvariancia ", vari,"\n desvio padrao", desv, "\nepocas", epocas)
plot_media_acerto(epocas, valores_acertos)


# In[ ]:


class Rotulo_com_distancia(object): 
    def __init__(self, padrao, saida_padrao, distancia): 
        self.__padrao = padrao 
        self.__distancia = distancia 
        self.__saida_padrao = saida_padrao

    def __repr__(self): 
        return "Rotulo:%s Saida padrão:%s Distancia:%s" % (self.__padrao, self.__saida_padrao, self.__distancia)

    def get_padrao(self): 
        return self.__padrao
    
    def get_saida_padrao(self):
        return self.__saida_padrao

    def get_distancia(self): 
        return self.__distancia


# In[ ]:


produtos = []
produto1 = Rotulo_com_distancia("chocolate", 1, 3.45)
produto2 = Rotulo_com_distancia("cacau",0, 0.0)
produtos.append(produto1)
produtos.append(produto2)
print (produtos) 


produtos_ordenados = sorted(produtos, key = Rotulo_com_distancia.get_distancia, reverse=True)
print(produtos_ordenados)


# In[ ]:


x = [1,2]
y = [2,4]

print(calcular_distancia(x, y))


# In[22]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
padrões para coluna vertebral para 2 classes
'''
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo, classe):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
        #verifica quantas classes quero separar os padroes    
        if (classe == 2):    
            if(cont <210):
                x[len(x)-1] = 1
            else:
                x[len(x)-1] = 0
        elif (classe == 3):
            if(cont < 50):
                x[len(x)-1] = 0
            elif (cont >= 50 and cont < 100):
                x[len(x)-1] = 0.5
            else:
                x[len(x)-1] = 1
        
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close() 
    
    for i in padroes:
        i.pop(1)
        i.pop(1)
    return padroes

c = ler_arquivo("column_2C.dat", 2)
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)
f = dados.tolist() #vetor de padroes da iris 


#print("****padroes*****\n", f, len(f))
#p1, p2 = coordenadas(w)
plotar_sem_reta(dados)

valores_acertos = []
#escolha a quantidade de classes a separar os padroes
valores_acertos, media, vari, desv, epocas = verificar_classe_padrao(f, 2) 

print("****valores dos acertos*****","\n", valores_acertos, "\nmedia", media, "\nvariancia ", vari,"\n desvio padrao", desv, "\nepocas", epocas)
plot_media_acerto(epocas, valores_acertos)


# In[ ]:


'''Data Set Information:

Biomedical data set built by Dr. Henrique da Mota during a medical residence period in the Group of Applied Research in Orthopaedics (GARO) of the Centre MÃ©dico-Chirurgical de RÃ©adaptation des Massues, Lyon, France. The data have been organized in two different but related classification tasks. The first task consists in classifying patients as belonging to one out of three categories: Normal (100 patients), Disk Hernia (60 patients) or Spondylolisthesis (150 patients). For the second task, the categories Disk Hernia and Spondylolisthesis were merged into a single category labelled as 'abnormal'. Thus, the second task consists in classifying patients as belonging to one out of two categories: Normal (100 patients) or Abnormal (210 patients). We provide files also for use within the WEKA environment.
'''


# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
padrões para coluna vertebral para 2 classes
'''
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo, classe):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
        #verifica quantas classes quero separar os padroes    
        if (classe == 2):    
            if(cont <210):
                x[len(x)-1] = 1
            else:
                x[len(x)-1] = 0
        elif (classe == 3):
            if(cont < 100):
                x[len(x)-1] = 0
            elif (cont >= 100 and cont < 160):
                x[len(x)-1] = 0.5
            else:
                x[len(x)-1] = 1
        
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close() 
    
    for i in padroes:
        i.pop(1)
        i.pop(1)
    return padroes

c = ler_arquivo("column_2C.dat", 3)
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)
f = dados.tolist() #vetor de padroes da iris 


#print("****padroes*****\n", f, len(f))
#p1, p2 = coordenadas(w)
plotar_sem_reta(dados)

valores_acertos = []
#escolha a quantidade de classes a separar os padroes
valores_acertos, media, vari, desv, epocas = verificar_classe_padrao(f, 3) 

print("****valores dos acertos*****","\n", valores_acertos, "\nmedia", media, "\nvariancia ", vari,"\n desvio padrao", desv, "\nepocas", epocas)
plot_media_acerto(epocas, valores_acertos)

