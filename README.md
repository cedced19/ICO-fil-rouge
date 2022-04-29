# Fil rouge

Yishan Sun, Simon Kurney, Pablo Aldana, Cédric Jung, Baptiste Deconihout, Zoé Poupardin

[Compte Rendu](CR.pdf)
[Présentation](Présentation.pdf)

## Organisation des fichiers

- Dans `agents/`, on retrouve les agents avec la méthode QLearning implémenté uniquement pour l'Algorithme Génétique
- Dans `agents_multithreads/`, on retrouve le SMA en version multithreads sans QLearning
- Dans `agents_multiprocess/`, on retrouve le SMA en version multicores avec QLearning implémenté pour les trois algorithmes (AG, RS, Tabou)
- Dans `mesa-tea/`, on retourve un TEA qui nous avait été demandé
- Dans `rs`, l'algoritme du recuit simulé
- Dans `rs_QL`, l'algoritme du recuit simulé avec QLearning
- Dans `tabou`, l'algoritme tabou
- Dans `tabou_QL`, l'algoritme tabou avec QLearning



Le dossier `agents_multiprocess/` est donc notre dossier final. 
On a dans ce dossier le fichier `AGentmodel.py` qui est notre fichier principal avec l'agent SMA.
Le fichier `big_example.py` peut être executer pour tester le tout avec un un exemple de 50 clients.