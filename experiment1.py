import pandas as pd
import numpy as np
from scan_wrapper import scan_wrapper
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

def read_spectrogram(testfile, gtfile):
	spectrogram = pd.read_csv(testfile, header=None)
	spectrogram = spectrogram.to_numpy()

	groundtruth = pd.read_csv(gtfile, header=None)
	groundtruth = groundtruth.to_numpy()

	return spectrogram, groundtruth

def get_testfile(snr, filenum):
	return f"snr_{snr}/test_file_{snr}_2_{snr}_{filenum}_.csv"

def get_gtfile(snr, filenum):
	return f"groundtruth/snr_{snr}/test_file_{snr}_2_{snr}_{filenum}_groundtruth_.csv"

if __name__=="__main__":
	vary_snr_prefix = "/Users/clarkmattoon/OneDrive - University at Albany - SUNY/docs/ubinetlab/tx_detection/vary_snr/"

	snrs = [-90, -95, -100, -102, -104, -105, -106]
	filenums = range(1, 101)
	# noisefloor = -109

	results = {snr:[] for snr in snrs}
	for snr in snrs:
		for filenum in filenums:
			testfile = vary_snr_prefix + get_testfile(snr, filenum)
			gtfile = vary_snr_prefix + get_gtfile(snr, filenum)
			spectrogram, groundtruth = read_spectrogram(testfile, gtfile)

			t1, f = groundtruth.shape
			t = 2**int(np.log2(t1))
			groundtruth = groundtruth[:t, :]

			omega, pred = scan_wrapper(spectrogram)

			total_characterization = np.zeros(pred[0].shape)

			for p in pred:
				total_characterization = np.logical_or(total_characterization, p, casting="unsafe").astype(int)

			accuracy = jaccard_score(total_characterization, groundtruth, average="micro")
			results[snr].append(accuracy)
			print(f"{snr=}, {filenum=}, {accuracy=}")

	mean_accuracy = [np.mean(results[snr]) for snr in snrs]

	plt.plot(snrs, mean_accuracy, "-o")
	plt.xlabel("Mean signal power (dBm)")
	plt.ylabel("Accuracy")
	plt.title("Characterization Accuracy Vs SNR")
	plt.show()
