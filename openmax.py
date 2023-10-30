
import libmr
import numpy as np
import scipy.spatial.distance as spd


def compute_mav_distances(activations, predictions, true_labels):
    """
    Calculates the mean activation vector (MAV) for each class and the distance to the mav for each vector.

    :param activations: logits for each image.
    :param predictions: predicted label for each image.
    :param true_labels: true label for each image.
    :return: MAV and euclidean-cosine distance to each vector.
    """
    correct_activations = list()
    mean_activations = list()
    eucos_dist = np.zeros(6)

    for cl in range(6):
        # Find correctly predicted samples and store activation vectors.
        i = (true_labels == predictions)
        i = i & (predictions == cl)
        act = activations[i, :]
        correct_activations.append(act)

        # Compute MAV for class.
        act = act.detach().numpy()
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute all, for this class, correctly classified images' distance to the MAV.
        for col in range(len(act)):
            eucos_dist[cl] += spd.euclidean(mean_act, act[col, :])/200. + spd.cosine(mean_act, act[col, :])

    return mean_activations, eucos_dist/len(act)


def weibull_tailfitting(eucos_dist, mean_activations, taillength=20):
    """
    Fits a Weibull model of the logit vectors farthest from the MAV.

    :param eucos_dist: the euclidean-cosine distance from the MAV.
    :param mean_activations: mean activation vector (MAV).
    :param taillength:
    :return: weibull model.
    """

    weibull_model = {}
    for cl in range(6):
        weibull_model[str(cl)] = {}
        weibull_model[str(cl)]['eucos_distances'] = eucos_dist[cl]
        weibull_model[str(cl)]['mean_vec'] = mean_activations[cl]
        weibull_model[str(cl)]['weibull_model'] = []
        mr = libmr.MR(verbose=True)
        tailtofit = sorted(mean_activations[0])[-taillength:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[str(cl)]['weibull_model'] = mr

    return weibull_model

def compute_open_max_probability(openmax_known_score, openmax_unknown_score):
    """
    Compute the OpenMax probability.

    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """

    prob_closed, prob_open, scores = [], [], []

    # Compute denominator for closet set + open set normalization.
    # Sum up the class scores.
    for category in range(6):
        scores += [np.exp(openmax_known_score[category])]
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)

    # Scores for image belonging to either closed or open set.
    prob_closed = np.array([scores / total_denominator])
    prob_open = np.array([np.exp(openmax_unknown_score) / total_denominator])

    probs = np.append(prob_closed.tolist(), prob_open)

    assert len(probs) == 7
    return probs


def recalibrate_scores(weibull_model, img_layer_act, alpharank=6):
    """
    Computes the OpenMax probabilities of an input image.

    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probabilities of image.
    """

    num_labels = 6

    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_labels)
    for i in range(0, len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):
        label_weibull = weibull_model[str(categoryid)]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[str(categoryid)]['mean_vec']    # Obtain MAV for specific class.
        img_dist = spd.euclidean(label_mav, img_layer_act)/200. + spd.cosine(label_mav, img_layer_act)

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[categoryid] * (1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [img_layer_act[categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)

    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab