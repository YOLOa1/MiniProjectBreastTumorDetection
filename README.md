# 🧠 Mini Projet - Breast Cancer Detection

Ce projet a pour objectif de détecter les tumeurs du sein à partir d’images médicales (DICOM, JPG, PNG). Il utilise un modèle de détection d’objets basé sur Faster R-CNN pour identifier et encadrer les zones tumorales dans une image.

## 📌 Description

À partir d’une ou plusieurs images d’entrée (formats DICOM, JPG ou PNG), le modèle génère une image annotée avec une boîte englobante délimitant la zone suspecte ou tumorale.  
Le projet est orienté vers une application en diagnostic assisté par intelligence artificielle (IA) dans le domaine de l'imagerie médicale.

---

## 🧰 Technologies utilisées

- Python 3.x
- PyTorch
- Torchvision
- pydicom
- NumPy

---

## 🗂️ Organisation du code

### `Model_Training.py`

Ce script comprend :

- 📁 **Chargement des données DICOM** : lecture des slices d’imagerie médicale et des masques de tumeurs.
- 🧠 **Création d’un dataset personnalisé** : `DicomTumorDataset` pour coupler les images et les masques.
- 🏗️ **Entraînement d’un modèle Faster R-CNN** pré-entraîné, adapté à notre tâche de détection binaire (présence ou absence de tumeur).
- 💾 **Sauvegarde du modèle entraîné** : le modèle est enregistré sous le nom `fasterrcnn_dicom_tumor.pth`.

---

## 🚀 Exécution


<div align="center">
  <table>
    <tr>
      <td align="center">
        <strong>Input</strong><br>
        <img src="Example%20of%20usage/Input.jpg" width="300" style="margin-right: 20px;"/>
      </td>
      <td align="center">
        <strong>Output</strong><br>
        <img src="Example%20of%20usage/Output.png" width="300" style="margin-left: 20px;"/>
      </td>
    </tr>
  </table>
</div>

