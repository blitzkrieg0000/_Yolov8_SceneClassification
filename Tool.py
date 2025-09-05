import numpy as np
# np.set_printoptions(suppress=True, precision=4)


def CalculateConfusionMatrix(confusion_matrix, labels=[], number_classes=None, transpoze=False) -> list[dict]:
    if not number_classes and not labels:
        number_classes = confusion_matrix.shape[0]
    elif labels:
        number_classes = len(labels)
    
    if transpoze:
        confusion_matrix = np.array(confusion_matrix).T
    results = []
    for c in range(number_classes):
        # Satırlar: ACTUAL X Sütunlar: PREDICTED
        label = labels[c] if labels else ''
        tp = confusion_matrix[c,c]
        fp = sum(confusion_matrix[:,c]) - confusion_matrix[c,c]
        fn = sum(confusion_matrix[c,:]) - confusion_matrix[c,c]
        tn = sum(np.delete(sum(confusion_matrix)-confusion_matrix[c,:],c))

        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        specificity = tn/(tn+fp)
        f1_score = 2*((precision*recall)/(precision+recall))
        fpr = fp/(fp+tn) #= 1-specificity

        actual_samples = int(fn + tp)
        predicted_samples = int(fp + tp)
        all_samples = int(confusion_matrix.sum().sum())

        results.append({
            "class": c,
            "label": label,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1" : f1_score,
            "fpr" : fpr,
            "all_samples" : all_samples,
            "actual_samples" : actual_samples,
            "predicted_samples" : predicted_samples
        })

        print(
            f"\nmetrics for class {c} - {label} =>\n\
            Recall: {round(recall,4)},\n\
            Specificity: {round(specificity,4)},\n\
            Precision: {round(precision,4)},\n\
            all_samples: {all_samples},\n,\
            actual_samples: {actual_samples},\n,\
            predicted_samples: {predicted_samples},\n\
        ")
    
    return results