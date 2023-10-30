import numpy as np
from utils import *
from libemg.utils import *
from libemg.emg_classifier import * 
from libemg.feature_extractor import FeatureExtractor 
from LSTM import LSTM
from AE import AE
from openmax import *
from sktime.transformations.panel.interpolate import TSInterpolator

WINDOW_SIZE = 30
WINDOW_INCREMENT = 10
FEATURES = ['MAV']

fix_random_seed(1)
subject = 2
fe = FeatureExtractor()

dataset_folder = 'Data/S' + str(subject) + '/ww_end/'
test_gestures = []
train_gestures = []
train_labels = []
test_gestures = []
test_labels = []
for c in ["1","2","3","4","5","6"]:
    for r in ["0","1","2","3","4","5","6","7","8","9"]:
        data = np.loadtxt(dataset_folder + c + '/R_' + r + '_C_0.csv', delimiter=',')
        windows = get_windows(data, WINDOW_SIZE, WINDOW_INCREMENT)
        gesture = EMGClassifier()._format_data(fe.extract_features(FEATURES, windows))
        if len(gesture) < 20:
            ts = TSInterpolator(20)
            gesture = ts.fit_transform(gesture)
        if int(r) > 7:
            test_labels.append(int(c)-1)
            test_gestures.append(gesture)
        else:
            train_labels.append(int(c)-1)
            train_gestures.append(gesture)

adl_windows = extract_adl_data(subject, WINDOW_SIZE, WINDOW_INCREMENT)[0]

dl = make_data_loader(np.array(train_gestures), np.array(train_labels))
# Gesture segmentation 
ae = AE(train_gestures[0].shape[1])
ae.fit(dl, num_epochs=50)

# Gesture prediction 
lstm = LSTM(n_classes=len(np.unique(train_labels)), n_features=8)
lstm.fit(dl, num_epochs=50)


# # Check test data
# t = torch.tensor(test_gestures, dtype=torch.float32)
# scores_nsm = lstm.forward_nosm(t)
# so, sm = openmax(weibull_model, categories, scores_nsm[0], 0.5, alpha=6)

train_d = torch.tensor(train_gestures, dtype=torch.float32)
scores_nsm = lstm.forward_nosm(train_d)
preds = [s.argmax().item() for s in scores_nsm]
mean_activations, eucos_dist = compute_mav_distances(scores_nsm, np.array(preds), np.array(train_labels))
print('Weibull tail fitting tail length: {}'.format(20))
weibull_models = weibull_tailfitting(eucos_dist, mean_activations, taillength=20)


# ------- First lets run on the test data ------------- # 
t = torch.tensor(test_gestures, dtype=torch.float32)
scores_nsm = lstm.forward_nosm(t)
test_probs_om = [recalibrate_scores(weibull_models, s.detach().numpy(), alpharank=6) for s in scores_nsm]
scores_sm = lstm.forward(t)
test_probs_sm = np.vstack(scores_sm.detach().numpy())


# ------- Second lets run on ADL data ------------- # 
adl_windows = extract_adl_data(subject, WINDOW_SIZE, WINDOW_INCREMENT)[0]
adl_features = EMGClassifier()._format_data(fe.extract_features(FEATURES, adl_windows))
adl_features = np.array([np.array(adl_features[z:z+20]) for z in range(0,len(adl_features) - 20)])
adl_f = torch.tensor(adl_features, dtype=torch.float32)
scores_nsm_adl = lstm.forward_nosm(adl_f)
scores_sm_adl = lstm.forward(adl_f)


print("Here")