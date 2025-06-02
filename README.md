# 🧠 Mini Projet - Breast Cancer Detection

Ce projet a pour objectif de détecter les tumeurs du sein à partir d’images médicales (DICOM, JPG, PNG). Il utilise un modèle de détection d’objets basé sur Faster R-CNN pour identifier et encadrer les zones tumorales dans une image.

## 📌 Description

À partir d’une ou plusieurs images d’entrée (formats DICOM, JPG ou PNG), le modèle génère une image annotée avec une boîte englobante délimitant la zone suspecte ou tumorale.  
Le projet est orienté vers une application en diagnostic assisté par intelligence artificielle (IA) dans le domaine de l'imagerie médicale.

---

## 🧰 Technologies utilisées

- Python 3.11
- PyTorch
- Torchvision
- pydicom
- NumPy

---

## Dataset

 <strong>L'accès au dataset à partie de ce lien</strong><br>
        <a href="https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/">
          Duke-Breast-Cancer-MRI
        </a>

        
 <strong>L'accès direct aux elements d'entrainement + model à partir de ce drive</strong><br>
        <a href="https://drive.google.com/drive/folders/1X_KR_CjcM160m_fgApnUabu-laRp1LoD?usp=drive_link">
          Drive
        </a>      


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

Exemple d'une visualisation sur un ensemble d'image hors entrainement : 

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

---


## 📬 Feedback & Contact</h2>
<p>Nous serions ravis d’avoir votre avis, vos suggestions ou vos questions sur ce projet ! N'hésitez pas à nous contacter sur LinkedIn :</p>
<ul>
  <li><a href="https://www.linkedin.com/in/adam-najid/" target="_blank" rel="noopener noreferrer">NAJID Adam</a></li>
  <li><a href="https://www.linkedin.com/in/romaissae-belhaj-635000360?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" rel="noopener noreferrer">BELHAJ Romaissae</a></li>
</ul>
<p>Ce projet a été réalisé dans le cadre de notre formation, sous l'encadrement de :</p>
<ul>
  <li>EL OSSMANI Mustapha</li>
  <li>BERRADA Mohammed</li>
</ul>
<p>Faites-nous savoir comment nous pourrions améliorer le projet ou collaborer avec vous !</p>

