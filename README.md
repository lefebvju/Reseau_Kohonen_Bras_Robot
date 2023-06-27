LEFEBVRE Julien, MERCIER Loris

# Techniques d'IA : Réseaux de neurones

## 3. Etude "théorique" de cas simple
### 3.1 Influence de η
Dans le cas, η=0: 
- La formule de mise à jour des poids sera nulle. Il n'y a donc pas d'apprentissage, les poids restent identiques.

Dans le cas, η=1:
- La mise à jour des poids n'est pondérée que par l'exponentielle. Le numérateur et le dénominateur de l'exponentielle étant positif, la valeur de fonction oscille entre 0 et 1.
Dans le cas du neurone gagnant, le numérateur sera nulle, donc e(..)=1.
La formule sera alors ∆wji=1\*1\*(xi - wji) = (xi - wji), c'est à dire une variation du poids équivalent à son entrée X moins son poids.
Le neurone gagnant étant celui ayant un poid proche de l'entrée X, le neurone gagnant aura une évolution de poids moins marquée que les autres neurones.
Il n'y a pas de convergence dans ce cas.

Dans le cas, 0<η<1:
- L'exponentielle reste bornée entre 0 et 1 pour les mêmes raisons qu'auparavant.  
Avec un η<1, la mise à jour des poids va donc se réduire de pas en pas. Dans le cas du neurone gagnant, l'exponentielle vaut 1, la formule est alors : ∆wji=η*(xi - wji). Autrement dit, le poids va évoluer de η multiplié par la distance entre W* et X. Si la distance vaut 1, et η=0.1, le poids évoluera de 0.1 fois la distance.

Dans le cas, η>1:
- Avec η=2, on multiplie la distance entre W* et X par un facteur 2, on va donc se retrouver de "l'autre côté" de X.  
Si la distance=1 et poids=0.5, le nouveau poids sera alors 1.5.


### 3.2 Influence de σ
Si σ augmente :
- Le dénominateur de l'exponentielle augmente, la fraction va donc diminuer et tendre vers 0. Plus la puissance de l'exponentielle négative est petite, plus la fonction est grande. Comme nous tendons vers 0, l'exponentielle va donc grandir. Les neurones proches du neurone gagnant vont donc plus apprendre de l'entrée courante.

A convergence :
- En reprenant la logique ci-dessus, les neurones proches du gagnant vont beaucoup apprendre et donc avoir une mise à jour des poids plus grande. Ils vont donc se rapprocher de ce dernier. A convergence, nous avons alors une représentation beaucoup plus serrée.

Formule mathématiques :
- On voudrait calculer **f(x) = la moyenne des écarts des voisins du graphe**.

### 3.3 Influence de la distribution d'entrée
Autant de X1 que de X2:
- Le neurone va apprendre autant de X1 que de X2. Avec un η faible, on va converger. Le poids va alors tendre vers ∆wji = (X1+X2)/2

- Le neurone va apprendre n fois de X1 que de X2. Avec un η faible, on va converger. Le poids va alors tendre vers ∆wji = (nX1+X2)/(n+1)

- On conserve un η faible, les poids des neurones évoluent donc de manière à se rapprocher pas à pas des valeurs d'entrée. En cas de forte densité, les valeurs d'entrée seront proches. Les poids vont donc tendre vers la même direction. Les neurones seront alors proches dans la représentation.


## 4.3 Analyse de l'algorithme
Nous commençons cette étude en reprenant nos hypothèses théoriques de la première section.  

### a. Taux d'apprentissage η
Dans un premier temps, nous fixons tous les paramètres sauf êta. (σ=1.4)

Pour η=0, nous avions fait l'hypothèse qu'il n'y avait pas d'apprentissage.  
En simulant l'expérience, nous observons ce même résultat. Cela est assez logique au vu de la formule de mise à jour des poids. A chaque pas, nous avons un produit entre le taux d'apprentissage η, une exponentielle influencée par la largeur du voisinnage sigma, et un vecteur de l'écart entre W et X. L'une de ces 3 expressions étant nulles, toute la formule est nulle. Les poids n'évoluent pas, le modèle reste donc fixe et n'apprend pas.

Pour η=1, nous avions fait l'hypothèse qu'il n'y aurait aucune convergence.  
En simulant l'expérience, on constate effectivement que les points ne se stabilisent pas. En revanche, ils prennent très vite une forme de grille. En testant avec un autre jeu de donnée, on remarque un comportement similaire. Les neurones se placent très vite "au bon endroit" mais oscillent de manière assez forte.  
A partir de ces constats, nous pouvons alors pousser notre hypothèse initiale. Avec η=1, le taux d'apprentissage est considéré comme "fort". Les données apprennent très vite expliquant ainsi leur disposition rapide en forme de grille (dans le cas d'un jeu de donnée distribué uniformément dans l'espace). Eta influe donc sur la vitesse d'apprentissage.

Avec η>1, on observe que les neurones apprennent très fortement, ils oscillent encore plus fort rendant impossible une stabilisation de l'apprentissage.

Avec 0<η<1, on peut confirmer nos conclusions ci-dessus. Plus η se rapproche de 0, plus le temps d'apprentissage est long. Les neurones sont néanmoins beaucoup plus stable. A l'inverse, plus η augmente, plus le temps d'apprentissage est rapide mais les neurone perdent leur stabilité.  
Cette notion de "stabilité" se rapporte plus formellement à la mise à jour du poids à chaque itération. Avec un η proche de 0, le nouveau poids est proche de l'ancien. Le décalage à chaque itération est donc faible. 

Dans une version plus élaborée de notre simulation, nous pourrions alors avoir un modèle avec un ETA fort au début pour apprendre vide, puis faible à la fin pour affiner les résultats et stabiliser l'apprentissage.


### b. Largeur du voisinage σ
On fixe désormais le paramètre σ.

**Le cas de base servant de "témoin" à l'expérience est :**
* η=0.05
* σ=1.4   
* Carte : carré
* Erreur quantification = 0.025
* Mesure d'auto-organisation = 0.15

L'erreur de quantification permet de mesurer à quel point notre grille de données recouvre l'ensemble de l'espace initial. Plus l'erreur est faible, mieux nous couvrons le spectre de donnée.

La mesure d'auto-organisation mesure la distance entre les poids de nos neurones. Plus elle est la petite, plus la proximité des neurones augmente. La grille recouvre alors moins d'espace, représentant ainsi moins bien les données.

=> En reprenant nos hypothèses initiales, on suppose que sigma agit sur la distance entre les voisins. A quel point un neurone apprend du gagant, et donc se "rapproche" de lui.

**En augmentant sigma :**
* η=0.05
* σ=5
* Erreur quantification = 0.25
* Mesure d'auto-organisation = 0.06

=> Avec un sigma fort, on observe une grille beaucoup plus resserrée. Elle couvre moins l'espace initial. Cela se confirme avec les valeurs que nous obtenons. L'erreur de quantification augmente, nous représentons donc moins bien l'espace de données. La mesure d'auto-organisation diminue signifiant un resserement des neurones entre-eux allant dans le sens de notre observation.

**En diminuant sigma :**
* η=0.05
* σ=0.5
* Erreur quantification = 0.009
* Mesure d'auto-organisation = 0.27  

=> Avec un sigma plus faible, on observe une grille beaucoup plus lâche. Les poids des neurones sont bien mieux répartis dans la grille. En revanche, la forme "carré" de la grille est moins bien respectée.
En regardant nos mesures, on note une erreur de quantification très faible. L'espace de la carte est donc très bien respectée ! L'auto organisation augmente indiquant des distances plus grandes entre les neurones. Cela parait plutôt logique dans le cadre d'un espace de données carré. Pour mieux couvrir l'espace, il faut écarter au maximum les neurones.

**Conclusion :**
Sigma agit donc bel et bien sur l'espacement entre les données. Plus sa valeur augmente, plus les neurones se mettent à jour en fonction du neurone "gagnant". Ils se rapprochent ainsi de lui, provoquant un resserement de la grille. A l'inverse, un sigma faible entraine une influence plus faible du neurone gagnant. La grille est donc beaucoup plus lâche recouvrant mieux l'espace de données initial.
Ces conclusions se confirment avec l'analyse de nos métrics : l'erreur de quantification diminue lorsque la grille recouvre au mieux l'espae de donnée. Dans le cas d'une topogie carré, avec des données réparties uniformément dans tout l'espace, notre mesure d'auto organisation est anti-corrélée avec l'erreur de quantification. Plus la grille recouvre l'espace, plus les neurones sont écartées. L'erreur diminue mais l'auto organisation augmente.


### c. Nombre de pas
On a émis l'hypothèse que N est lié au nombre de fois que le réseau de neurones itère sur les données, ce qui signifie que le modèle aura un taux d'erreur plus faible en augmentant N.

On fixe :
* η=0.05
* σ=1.4

Nos résultats :
|   N   | Erreur quantitative | Mesure d'auto-organisation |
|:-----:|:-------------------:|:--------------------------:|
|  50   |        0.20         |            0.40            |
|  500  |        0.057        |            0.21            |
| 5000  |        0.024        |            0.15            |        
| 15000 |        0.021        |            0.16            |
| 30000 |        0.023        |            0.15            |

Nos résultats corroborent en partie avec l'hypothèse. En augmentant N, l'erreur diminue. On remarque toutefois une stabilisation des résultats à partir du pas 5000. En effet, à partir du moment où les poids des neurones "convergent", le nombre de pas n'a plus réellement d'impact. Du pas 5000, à 15000 puis 30000 les résultats sont semblables car la mise à jours des poids évoluent très peu.


### d. Taille et forme de la carte
Nous avons testé 3 formes (carré, rectangle, ligne) et 2 tailles différentes (15 neurones, 250 neurones).  
Notre hypothèse est que les formes en lien avec la disposition des données initiales obtiennent des meilleurs résultats. Nos données étant uniformément réparties dans l'espace, une forme de grille carré semble être la plus appropriée.

Nous conservons nos paramètres initiaux pour ce test :
* η=0.05
* σ=1.4
* Carte = carré 
* Données uniformément réparties.

|      Forme       | Erreur quantitative | Mesure d'auto-organisation |
|:----------------:|:-------------------:|:--------------------------:|
|    carré(5,5)    |        0.10         |            0.23            |
|   carré(15,15)   |        0.011        |            0.12            |
|  rectangle(4,6)  |        0.125        |            0.23            |
| rectangle(25,10) |        0.014        |            0.13            |
|   ligne(1,25)    |        0.060        |            0.20            |
|   ligne(1,250)   |        0.007        |            0.10            |

*Note :  Pour les lignes, nous avons du augmenter de façon significative le nombre de pas d'apprentissage.*

Visuellement, nos résultats confirment notre hypothèse. Les formes carrées couvrent l'espace au mieux tandis que le rectangle crée quelques espaces vides. La ligne à quant à elle beaucoup plus de mal à couvrir l'espace de données. De plus, le temps d'apprentissage de cette forme est très longue. Il faut le temps que la ligne se "déplie", ce qui n'est pas optimal.

Quantitativement, nos résultats sont légèrement différents. La ligne obtient de très bons résultats ! Cela peut se comprendre vis-à-vis de sa forme. Certes, elle ne couvre pas tout l'espace, mais l'espace qui l'est est très bien couvert. En moyenne, les résultats sont donc plutôt corrects. Il faudrait s'interroger sur des "sous-partie" de la carte pour constater à quel point la répartition est bonne sur certains points mais très mauvaise sur d'autres.  
Nos résultats quantitatifs nous en apprennent plus sur l'importance de la taille de la carte. Plus la taille est grande, plus les résultats sont bons. Logique, avec plus de points, nous convrons mieux l'espace. Dans la théorie, il faudrait une taille de carte égale au nombre de données.



### e. Jeu de données
Nous pensons qu'en cas de jeu de données non uniformément distribués, nos neurones vont se répartir en fonction de la distribution. Plus une partie de l'espace aura de données, plus le nombre de neurones sur cette partie sera grande.

Nous testons avec un carré (15,15).
Nous conservons les paramètres itinitiaux.

Nous utilisons l'ensemble de données 3 pour le témoin (T), puis ce même schéma (E) pondéré avec 1/10 des données réparties en haut à gauche et 9/10 en bas à droite.

| Répartition |   erreur quantitative   |  Mesure d'auto-organisation   |
|:-----------:|:-----------------------:|:-----------------------------:|
|     T       |          0.007          |             0.11              |
|     E       |          0.006          |             0.09              |

Visuellement, les résultats confirment notre hypothèse. La grille se déforme plus dans (E) avec beaucoup plus de point cette partie de la répresentation (9/10 des données se trouve ici).

Quantitativement, l'erreur est similaire. Cela signifie que notre grille couvre aussi bien l'espace dans les deux cas. Elle prend donc bien en compte la distribution des données.


### 4.4 Bras robotique
#### Question 1 :
Pour prédire la position qu’aura le bras étant donné une position motrice, nous pouvons :
* Soit trouver le neurone étant le plus proche des poids (θ1, θ2) dans la carte motrice et retourner ses valeur de poids (x1,x2) de la deuxième carte. (méthode 1)
* Soit trouver les 4 neurones les plus proches, calculer leurs distances et la réappliquer sur les poids (x1,x2) de chaque neurones. (méthode 2)

Nous avons implémenté les deux méthodes et nous avons obtenu de meilleurs résultats avec la methode des quatre neurones.  
Sachant qu'une carte de Kohonen forme une grille, les valeurs entrées en paramètre sont toujours entre 4 neurones sauf dans le cas où les données sont en dehors de la grille. Dans ce dernier cas, la prédiction à peu de chance d'être correcte.  
En prenant le cas où les données d'entrée sont dans la grille, il suffit de pondérer les poids (x1,x2) de chaque neurone par l'inverse de la distance entre les données d'entrée et les poids (θ1, θ2) de chaque neurones.

* Pour prédire la position motrice étant donné une position spatiale que l’on souhaite atteindre, nous pouvons réaliser
  la même chose que précédemment en cherchant dans les poids (x1,x2) pour obtenir en résultat des poids (θ1, θ2).

#### Question 2 :

Le modèle de la carte auto-organisatrice de Kohonen se rapproche du modèle de gaz neuronal de Martinetz et Schulten. En effet, ces deux modèles possèdent de nombreuses similiarités comme le type d'apprentissage (non-supervisé), la fonction d'agrégation (la distance) ou encore des propriétes communes comme la quantification vectorielle.
Ces deux algorithmes utilisent également un partionnement de l'espace d'entrée avec les poids comme valeur.

Contrairement à la carte de Kohonen qui doit posséder une carte compatible à la distribution des poids en entrée de son algorithme, le gaz neuronal à l'avantage de ne pas dépendre de ces positions. Il n'y a pas de notion de comptatbilité entre la forme de la carte et les données vu que l'algorithme s'adapte directement aux valeurs des poids initiaux.

Le défaut du gaz neuronal est en revanche sa prise en compte du bruit. L'algorithme apprenant tout l'espace de données, cela signifie qu'il apprend aussi le bruit qui s'y trouve pouvant ainsi détériorer les performances du modèle.

#### Question 3 :

Afin de prédire uniquement la position spatiale à partir de celle de la motrice, nous aurions pu utiliser le Perceptron Multi-Couche. Cet algorithme est une excellente méthode pour trouver une sortie Y à partir d'une entrée X spécifiée.

L'avantage de cette autre méthode est de pouvoir correctement prendre en compte des données multi-dimensionnelles (ici doublet en X et en Y). 

L'algorithme est en revanche supervisé nous obligeant à étiqueter convenablement les données de tests afin d'obtenir des résultats concluant. Selon le nombre de couches utilisé par le modèle, l'algorithme peut aussi être plus lent que la carte de kohonen.

#### Question 4 :

Pour effectuer des prédictions, nous avons calculé le résultat en utilisant la pondération des quatre plus proches voisins en fonction de leur distance par rapport aux données d'entrée.
Nous avons constaté que cette méthode donne de meilleurs résultats que la simple utilisation du neurone le plus
proche.
Cependant, nous nous sommes rendus compte en lisant cette question que cette méthode ne peut pas être utilisée pour
trouver le chemin le plus court.
Nous allons donc utiliser le neurone le plus proche pour trouver le chemin le plus court. (méthode 1)
  
En partant de ce neurone, on calcule les distances entre les poids de tous ces voisins et les poids cibles. On 
ajoute ensuite à une liste le neurone qui nous rapproche le plus de l'arrivée, puis on itère jusqu'à arriver au neurone le plus 
de (θ'1, θ'2).

Nous obtenons le chemin entre le neurone le plus proche de (θ1, θ2) et le neurone le plus proche de (θ'1, θ'2). Pour 
chaque de ce chemin, nous pouvons extraire (x1,x2) pour connaitre la suite des positions spatiales prise par la main.
![image du chemin généré](http://jlefebvre.fr/M1/mif16/TIA_NN.png)