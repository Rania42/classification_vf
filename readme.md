# 🏦 Système de Classification de Documents Bancaires (PFE)

Ce projet implémente une solution de classification automatique d'images de documents bancaires et juridiques. Il utilise une approche **multimodale** combinant la vision par ordinateur et le traitement du langage naturel (NLP) pour une précision optimale.

---

## 🛠️ Architecture Technique

Le pipeline de classification repose sur l'intégration de plusieurs technologies de pointe :
* **Vision :** Extraction de features visuelles via **EfficientNet**.
* **Texte :** Traitement sémantique du contenu textuel (OCR) avec **mBERT** (Multilingual BERT).
* **Agents d'IA :** Analyse intelligente et raisonnement local avec **Llama 3.2** et **Qwen2.5-VL**.

> **Note sur la confidentialité :** Ce projet est conçu pour une exécution **locale** via Ollama, garantissant que les documents sensibles ne sont jamais envoyés sur des serveurs cloud externes.

---

## 📋 Prérequis

Avant de commencer, assurez-vous d'avoir installé :
* **Python 3.8** ou une version supérieure.
* **pip** (gestionnaire de paquets Python).
* **Ollama** (pour l'exécution locale des agents d'IA).

---

## 🚀 Installation

### 1. Cloner le repository
```bash
git clone [https://github.com/Rania42/classification_vf.git](https://github.com/Rania42/classification_vf.git)
cd classification_vf
2. Installer les dépendances
Bash

pip install -r requirements.txt
3. Configurer le modèle pré-entraîné
Le modèle bank_doc_classifier_multimodal.pth est trop volumineux pour être hébergé directement sur GitHub.

Téléchargez le modèle depuis ce lien Google Drive.

Créez un dossier nommé model/ à la racine du projet s'il n'existe pas.

Placez le fichier .pth à l'intérieur du dossier : model/bank_doc_classifier_multimodal.pth.

💻 Exécution de l'application
Pour lancer l'interface utilisateur web :

Bash

python app.py
Une fois lancé, l'application est accessible à l'adresse suivante : http://localhost:5000

🧪 Test rapide (Mode Sandbox)
Si vous souhaitez tester uniquement les performances du modèle de classification sans configurer l'application complète :

Ouvrez le fichier multimodale_test.ipynb dans Google Colab.

Exécutez toutes les cellules du notebook.

La dernière cellule vous permettra d'uploader une image pour obtenir une prédiction immédiate.

📁 Structure du projet
Plaintext

classification_vf/
├── app.py                  # Point d'entrée de l'application Flask
├── index.html              # Interface utilisateur (Frontend)
├── requirements.txt        # Liste des bibliothèques Python nécessaires
├── model_info.md           # Documentation technique du modèle
├── multimodale_test.ipynb  # Notebook pour tests unitaires
├── model/                  # Stockage local du modèle .pth
└── stored_documents/       # Dossier local pour les documents archivés
📄 Licence
Ce projet est réalisé dans le cadre d'un Projet de Fin d'Études (PFE).