# projet-big-data

## Ce qu'on a fait

- Web app 
    - Upload d'une image
    - Prédiction d'une image
    - Envoie de feedback
- Serving (backend fast api)
    - Entraînement
    - Prédiction
    - Réception de feedback et enregistrement dans prod_data.csv
    - Ré-entraînement et sauvegarde du nouveau modèle (dans `/artifacts/best_model.pkl`)

## Lancer serving

```shell
docker compose up --build --force-recreate
```

⚠️ Ca va mettre assez longtemps à résoudre les dépendances avec pip car on utilise pytorch et pytorch vision

💡 L'entraînement débute automatiquement au lancement de l'API avec les données du fichier `/data/ref_data.csv`

## Lancer webapp

Pareil 

```shell
docker compose up --build --force-recreate
```

# Demo

Si c'est trop long chez vous de build l'image `serving`, voici une demo

![bande annonce projet](Bande%20Annonce%20projet.gif)

(la vidéo se trouve [ici](Bande%20Annonce%20projet.mp4))