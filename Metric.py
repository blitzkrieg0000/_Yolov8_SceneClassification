import os
from itertools import combinations, cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer



RESULT_PATH = "./results/classification_train"

ground_y = np.load(os.path.join(RESULT_PATH, "real.npy"))
pred_y = np.load(os.path.join(RESULT_PATH, "predicted.npy"))
class_labels = np.load(os.path.join(RESULT_PATH, "labels.npy"), allow_pickle=True)
class_labels = list(np.array(class_labels).tolist().values())


label_binarizer = LabelBinarizer().fit(ground_y)
y_onehot_pred = label_binarizer.transform(pred_y)
y_onehot_ground = label_binarizer.transform(ground_y)
n_classes = len(label_binarizer.classes_)

#! Sınıf Başına Roc
# class_of_interest = 1
# class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
# RocCurveDisplay.from_predictions(
#     y_onehot_ground[:, class_id],
#     y_onehot_pred[:, class_id],
#     name=f"{class_of_interest} vs the rest",
#     color="darkorange"
# )


# =================================================================================================================== #
#! ROC Eğrisini "One vs All-Rest(oVr)"  yaklaşımı ile hesaplama                                                       #
# =================================================================================================================== #
"""
    Bire karşı hepsine karşı olarak da bilinen Bire Karşı Geriye Kalan (OvR) çoklu sınıf stratejisi, 
    n_sınıfların her biri için bir ROC eğrisinin hesaplanmasından oluşur.
    Her adımda, belirli bir sınıf pozitif sınıf olarak kabul edilir ve geri kalan sınıflar toplu olarak 
    negatif sınıf olarak kabul edilir.
"""

#! Micro-averaged One-vs-Rest ROC(Receiver Operating Characteristic)---------------------------------------------------
# Graph
RocCurveDisplay.from_predictions(
    y_onehot_ground.ravel(),
    y_onehot_pred.ravel(),
    name="micro-average OvR",
    color="darkorange"
)

# Value
micro_roc_auc_ovr = roc_auc_score(
    y_onehot_ground,
    y_onehot_pred,
    multi_class="ovr",
    average="micro",
)
print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.2f}")

#! Compute all micro-average ROC curve and ROC area--------------------------------------------------------------------
# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()

fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_ground.ravel(), y_onehot_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves")
plt.legend()
plt.show()


#! Compute OvR MACRO Average-------------------------------------------------------------------------------------------
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_ground[:, i], y_onehot_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

#* Yukarıdakini Kısaca (oVr - makro) çıkarmak için---------------------------------------------------------------------
macro_roc_auc_ovr = roc_auc_score(
    y_onehot_ground,
    y_onehot_pred,
    multi_class="ovr",
    average="macro",
)
print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")


#! TÜM ROC OvR Yaklaşımlı Curveler için----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_ground[:, class_id],
        y_onehot_pred[:, class_id],
        # name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
        # plot_chance_level=(class_id == 2),
    )

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Receiver Operating Characteristic(ROC) \nto One-vs-Rest multiclass")
plt.legend()
plt.show()

print()



# =================================================================================================================== #
#! ROC Eğrisini "One vs One(oVo)" yaklaşımı ile hesaplama                                                             #
# =================================================================================================================== #
"""
    Bire Karşı Bir (OvO) çoklu sınıf stratejisi, sınıf çifti başına bir sınıflandırıcının yerleştirilmesinden oluşur. 
    N_classes * (n_classes - 1) / 2 sınıflandırıcılarının eğitilmesini gerektirdiğinden, bu yöntem O(n_classes ~2) 
    karmaşıklığından dolayı genellikle One-vs-Rest'ten daha yavaştır.
"""

# ROC curve using the OvO macro-average--------------------------------------------------------------------------------
"""
    OvO şemasında ilk adım, tüm olası benzersiz çift kombinasyonlarını tanımlamaktır.
    Puanların hesaplanması, belirli bir çiftteki öğelerden birinin pozitif sınıf ve diğer öğenin negatif sınıf olarak ele alınması,
    ardından rollerin ters çevrilmesi ve her iki puanın ortalamasının alınması yoluyla puanın yeniden hesaplanmasıyla yapılır.
"""

import sys
sys.exit()

pair_list = list(combinations(label_binarizer.classes_, 2))
# print(pair_list)

pair_scores = []
mean_tpr = dict()

for ix, (label_a, label_b) in enumerate(pair_list):
    a_mask = y_onehot_ground == label_a
    b_mask = y_onehot_ground == label_b
    ab_mask = np.logical_or(a_mask, b_mask)

    a_true = a_mask[ab_mask]
    b_true = b_mask[ab_mask]

    idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
    idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

    fpr_a, tpr_a, _ = roc_curve(a_true, pred_y[ab_mask, idx_a])
    fpr_b, tpr_b, _ = roc_curve(b_true, pred_y[ab_mask, idx_b])

    mean_tpr[ix] = np.zeros_like(fpr_grid)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
    mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
    mean_tpr[ix] /= 2
    mean_score = auc(fpr_grid, mean_tpr[ix])
    pair_scores.append(mean_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        fpr_grid,
        mean_tpr[ix],
        label=f"Mean {label_a} vs {label_b} (AUC = {mean_score :.2f})",
        linestyle=":",
        linewidth=4,
    )
    RocCurveDisplay.from_predictions(
        a_true,
        pred_y[ab_mask, idx_a],
        ax=ax,
        name=f"{label_a} as positive class",
    )
    RocCurveDisplay.from_predictions(
        b_true,
        pred_y[ab_mask, idx_b],
        ax=ax,
        name=f"{label_b} as positive class",
        plot_chance_level=True,
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{class_labels[idx_a]} vs {label_b} ROC curves")
    plt.legend()
    plt.show()












