import tensorflow as tf
import itertools
import datetime
import zipfile
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Lit une image depuis filename, la convertit en tenseur et la redimensionne en
    (img_shape, img_shape, 3).

    Args:
        filename (str): chemin vers l'image cible
        img_shape (int): taille cible, default 224
        scale (bool): normaliser les pixels entre 0 et 1, default True
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img / 255.
    return img


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """
    Affiche une matrice de confusion annotée.

    Args:
        y_true: labels réels (même forme que y_pred)
        y_pred: labels prédits (même forme que y_true)
        classes: noms de classes sous forme de liste. Si None, des entiers sont utilisés.
        figsize: taille de la figure (default=(10, 10))
        text_size: taille du texte dans les cases (default=15)
        norm: afficher les valeurs normalisées (default=False)
        savefig: sauvegarder la figure en PNG (default=False)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(cm.shape[0])

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    if savefig:
        fig.savefig("confusion_matrix.png")


def pred_and_plot(model, filename, class_names):
    """
    Charge une image, effectue une prédiction avec le modèle et affiche
    l'image avec la classe prédite en titre.
    """
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Crée un callback TensorBoard et enregistre les logs dans :
    dir_name/experiment_name/<datetime>/

    Args:
        dir_name: répertoire cible pour les logs
        experiment_name: nom de l'expérience (ex: mobilenetv2_run1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def plot_loss_curves(history):
    """
    Trace les courbes de loss et d'accuracy (train vs validation) séparément.

    Args:
        history: objet History retourné par model.fit()
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(loss))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compare deux objets History TensorFlow sur un même graphique,
    utile pour visualiser une phase de fine-tuning après un entraînement initial.

    Args:
        original_history: History de la phase 1 (avant fine-tuning)
        new_history: History de la phase 2 (après fine-tuning)
        initial_epochs: nombre d'epochs de la phase 1
    """
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def unzip_data(filename):
    """
    Décompresse filename dans le répertoire courant.

    Args:
        filename (str): chemin vers le fichier zip à extraire
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def walk_through_dir(dir_path):
    """
    Parcourt dir_path et affiche le nombre de sous-dossiers et d'images par niveau.

    Args:
        dir_path (str): répertoire cible
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def calculate_results(y_true, y_pred):
    """
    Calcule accuracy, precision, recall et F1-score pour un modèle de classification binaire.

    Args:
        y_true: labels réels sous forme de tableau 1D
        y_pred: labels prédits sous forme de tableau 1D

    Returns:
        dict avec les clés accuracy, precision, recall, f1
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1,
    }
