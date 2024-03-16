# pomocnicza funkcja
def run(f_classify, x):
    return list(map(f_classify, x))

def evaluate(predictions, actual):
    correct = list(filter(
        lambda item: item[0] == item[1],
        list(zip(predictions, actual))
    ))
    return f"Accuracy {len(correct)/len(actual)*100:.0f} %"
