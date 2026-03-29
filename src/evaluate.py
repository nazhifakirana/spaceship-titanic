from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_val, y_pred))

    return acc