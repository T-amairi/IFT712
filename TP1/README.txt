Chaimae TOUBALI 
CIP : touc1402

Tahar AMAIRI
CIP : amat0601

Corentin POMMELEC
CIP : pomc0601

Nous avons modifié les trois fichiers du projet (notamment gestion_donnees.py). Par conséquent, pour une bonne
exécution, veuillez tester les fichiers ensemble. Concernant la recherche d'hyper-paramètre, il faut en premier 
lancer la fonction afin de déterminer une bonne valeur pour la variable "maxErrorDifference" permettant l'arrêt
de la recherche. En effet, si la différence d'erreur entre deux runs dépasse cette variable, alors la recherche
s'arrête. De cette manière, cela permet de stopper la recherche lorsqu'elle stagne et boucle à l'infini d'où un 
premier test afin de fixer "maxErrorDifference" à l'aide des différents logs. Par ailleurs, une alternative était
aussi de fixer le nombre d'itération maximale pour la recherche afin d'éviter de boucler à l'infini.