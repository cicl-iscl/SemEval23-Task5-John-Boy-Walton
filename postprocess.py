import os
from transformers import EvalPrediction
import numpy as np
# from scipy.special import softmax
import re
from collections import OrderedDict
import json


def postprocess_qa_predictions(
    dataset,
    predictions,
    n_best_size,
    max_answer_length,
    top_k=None,
    output_dir = None,
    prefix = 'eval'
):
    # """
    # Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    # original contexts. This is the base postprocessing functions for models that only return start and end logits.
    # Args:
    #     examples: The non-preprocessed dataset (see the main script for more information).
    #     features: The processed dataset (see the main script for more information).
    #     predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
    #         The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
    #         first dimension must match the number of elements of :obj:`features`.
    #     version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
    #         Whether or not the underlying dataset contains examples with no answers.
    #     n_best_size (:obj:`int`, `optional`, defaults to 20):
    #         The total number of n-best predictions to generate when looking for an answer.
    #     max_answer_length (:obj:`int`, `optional`, defaults to 30):
    #         The maximum length of an answer that can be generated. This is needed because the start and end predictions
    #         are not conditioned on one another.
    #     null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
    #         The threshold used to select the null answer: if the best answer has a score that is less than the score of
    #         the null answer minus this threshold, the null answer is selected for this example (note that the score of
    #         the null answer for an example giving several features is the minimum of the scores for the null answer on
    #         each feature: all features must be aligned on the fact they `want` to predict a null answer).
    #         Only useful when :obj:`version_2_with_negative` is :obj:`True`.
    #     output_dir (:obj:`str`, `optional`):
    #         If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
    #         :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
    #         answers, are saved in `output_dir`.
    #     prefix (:obj:`str`, `optional`):
    #         If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
    #     is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
    #         Whether this process is the main process or not (used to determine if logging/saves should be done).
    # """
    # assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    
    meaningful_pattern = re.compile('(\w|\d)+')
    bos_residuals = re.compile('(?<=^)(\W|\W)+')

    all_start_logits, all_end_logits = predictions

    # assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    # example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    # features_per_example = collections.defaultdict(list)
    # for i, feature in enumerate(features):
    #     features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = OrderedDict()



    all_nbest_json = OrderedDict()
    # if version_2_with_negative:
    #     scores_diff_json = collections.OrderedDict()

    # Logging.
    # logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    # logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for start_logits, end_logits, entry in zip(all_start_logits, all_end_logits, dataset):
        # Those are the indices of the features associated to the current example.
        # feature_indices = features_per_example[idx]

        # min_null_prediction = None
        prelim_predictions = []

        # # Looping through all the features associated to the current example.
        # for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
        # start_logits = all_start_logits[idx]
        # end_logits = all_end_logits[idx]
        # This is what will allow us to map some the positions in our logits to span of texts in the original
        # context.
        offset = entry["offset_mapping"]
        offset_starts, offset_ends = [pair[0] for pair in offset], [pair[1] for pair in offset]
        # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
        # available in the current feature.
        # token_is_max_context = features[feature_index].get("token_is_max_context", None)

        # Update minimum null prediction.
        # feature_null_score = start_logits[0] + end_logits[0]
        # if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
        #     min_null_prediction = {
        #         "offsets": (0, 0),
        #         "score": feature_null_score,
        #         "start_logit": start_logits[0],
        #         "end_logit": end_logits[0],
        #     }

        # go through n_best_size greater logits (from -1 to -n_best_size - 1 with step -1)
        start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    # start_index >= len(offset_mapping)
                    # or end_index >= len(offset_mapping)
                    offset[start_index] is None
                    or offset[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if (end_index < start_index) or (end_index - start_index + 1 > max_answer_length):
                    continue
                # Don't consider answer that don't have the maximum context available (if such information is
                # provided).
                # if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                #     continue
                prelim_predictions.append(
                    {
                        "offsets": (offset_starts[start_index], offset_ends[end_index]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )
        # if version_2_with_negative:
        #     # Add the minimum null prediction
        #     prelim_predictions.append(min_null_prediction)
        #     null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        prelim_predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        # if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
        #     predictions.append(min_null_prediction)

        predictions = []

        # Use the offsets to gather the answer text in the original context.
        context = entry["context"]
        for pred in prelim_predictions:
            offsets = pred.pop("offsets")
            text = context[offsets[0]: offsets[1]]
            # heuristic for cleaning the empty preds
            if len(meaningful_pattern.findall(text)):
                # heuristic for cleaning the non-empty preds
                text = bos_residuals.sub('', text)
                pred['text'] = text
                predictions.append(pred)

        # append 'smoothing' pred in case we have no non-null ones
        if not len(predictions):
            predictions.append({"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        # if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
        # if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
        #     predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # probs = softmax(
        #     [pred.pop("score") for pred in predictions]
        # )

        # Include the probabilities in our predictions.
        # for prob, pred in zip(probs, predictions):
        #     pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        # if not version_2_with_negative:
        if isinstance(top_k, int): best_predictions = predictions[:top_k]
        else: best_predictions = predictions[:1]
        all_predictions[entry["id"]] = [
            prediction["text"]
            for prediction in best_predictions
        ]

        # else:
        #     # Otherwise we first need to find the best non-empty prediction.
        #     i = 0
        #     while predictions[i]["text"] == "":
        #         i += 1
        #     best_non_null_pred = predictions[i]

        #     # Then we compare to the null prediction using the threshold.
        #     score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
        #     scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
        #     if score_diff > null_score_diff_threshold:
        #         all_predictions[example["id"]] = ""
        #     else:
        #         all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[entry["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]


    try:
        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

            prediction_file = os.path.join(
                output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
            )
            nbest_file = os.path.join(
                output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
            )

            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    except: pass

    return all_predictions

# Post-processing:
def postprocess(dataset, predictions, **kwargs): #, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        # examples=examples,
        dataset=dataset,
        predictions=predictions,
        **kwargs
        # null_score_diff_threshold=data_args.null_score_diff_threshold,
        # output_dir=training_args.output_dir,
        # is_world_process_zero=trainer.is_world_process_zero(),
        # prefix=stage,
    )
    # Format the result to the format the metric expects.
    # if data_args.version_2_with_negative:
    #     formatted_predictions = [
    #         {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    #     ]
    # else:
    formatted_predictions = [{"id": k, "prediction_texts": v} for k, v in predictions.items()]

    references = [{"id": entry["id"], "answers": entry['answers']} for entry in dataset]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)