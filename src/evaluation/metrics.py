import json

def exact_match(
    prediction,
    target
):
    return int(prediction == target)

def edit_distance(prediction, target):
    previous = list(range(len(target) + 1))
    for i, predicted_char in enumerate(prediction, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            replacement_cost = int(predicted_char != target_char)
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + replacement_cost)
            )
        previous = current

    return previous[-1]

def calculate_text_metrics(predictions, targets):
    if not predictions or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal nonzero lengths")

    total_errors = 0
    total_characters = 0
    valid_json = 0
    exact_matches = 0

    for prediction, target in zip(predictions, targets):
        total_errors += edit_distance(prediction, target)
        total_characters += len(target)
        target_object = json.loads(target)

        try:
            predicted_object = json.loads(prediction)
        except ValueError:
            continue

        valid_json += 1

        if predicted_object == target_object:
            exact_matches += 1

    count = len(predictions)

    return {
        "cer": total_errors / max(total_characters, 1),
        "json_valid_rate": valid_json / count,
        "json_exact_match": exact_matches / count
    }