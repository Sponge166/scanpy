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






	# multi_detects = {
	# 	-90: [], 
	# 	-95: [], 
	# 	-100: [4, 10, 12, 21, 22, 31, 47, 63, 64, 68, 76, 86], 
	# 	-102: [3, 5, 9, 10, 28, 30, 33, 34, 35, 40, 44, 49, 52, 54, 55, 58, 71, 72, 73, 78, 79, 80, 81, 82, 83, 87, 90, 95, 98, 100], 
	# 	-104: [1, 3, 4, 6, 7, 10, 11, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 36, 38, 39, 40, 42, 43, 45, 46, 47, 52, 55, 58, 62, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 79, 83, 85, 87, 88, 89, 90, 92, 96, 99, 100], 
	# 	-105: [2, 5, 7, 8, 10, 11, 15, 16, 19, 21, 22, 23, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 47, 48, 49, 50, 53, 54, 56, 57, 58, 60, 63, 64, 65, 66, 67, 69, 71, 72, 75, 76, 77, 78, 79, 81, 85, 86, 87, 88, 92, 93, 98, 99], 
	# 	-106: [2, 5, 6, 7, 9, 12, 14, 15, 17, 19, 20, 22, 25, 26, 27, 28, 29, 30, 31, 34, 35, 38, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 57, 58, 59, 61, 62, 63, 64, 65, 67, 70, 71, 72, 75, 76, 77, 78, 79, 84, 85, 86, 87, 89, 90, 92, 94, 96, 97, 99]
	# }

	# # print(sum([len(multi_detects[snr]) for snr in snrs]))

	# for snr in snrs:
	# 	if snr > -106:
	# 		continue
	# 	for filenum in multi_detects[snr]:
	# 		if filenum < 64:
	# 			continue
	# 		testfile = vary_snr_prefix + get_testfile(snr, filenum)
	# 		gtfile = vary_snr_prefix + get_gtfile(snr, filenum)
	# 		spectrogram, groundtruth = read_spectrogram(testfile, gtfile)

	# 		t1, f = groundtruth.shape
	# 		t = 2**int(np.log2(t1))
	# 		groundtruth = groundtruth[:t, :]

	# 		omega, pred = scan_wrapper(spectrogram)

	# 		# total_characterization = np.zeros(pred[0].shape)
	# 		accuracies = []
	# 		for p in pred:
	# 			# total_characterization = np.logical_or(total_characterization, p, casting="unsafe").astype(int)
	# 			accuracies.append(jaccard_score(p.astype(int), groundtruth, average="micro"))
	# 		max_accuracy = max(accuracies)
	# 		# accuracy = jaccard_score(total_characterization, groundtruth, average="micro")
	# 		print(f"{snr=}, {filenum=}, {max_accuracy=}")

	# 		if max_accuracy != 1:

	# 			total_characterization = np.zeros(pred[0].shape)
	# 			for p in pred:
	# 				total_characterization = np.logical_or(total_characterization, p, casting="unsafe").astype(int)

	# 			fig = plt.figure()
	# 			ax = fig.add_subplot(131)
	# 			plt.imshow(groundtruth, cmap=plt.cm.spring, aspect='auto')
	# 			plt.xlabel("Frequency [MHz]")
	# 			plt.ylabel("Time [ms]")
	# 			plt.title("Groundtruth")

	# 			ax = fig.add_subplot(132)
	# 			plt.imshow(total_characterization, cmap=plt.cm.spring, aspect='auto')
	# 			plt.xlabel("Frequency [MHz]")
	# 			plt.ylabel("Time [ms]")
	# 			plt.title("Total Characterization")

	# 			ax = fig.add_subplot(133)
	# 			plt.imshow(spectrogram[:t,:], cmap=plt.cm.spring, aspect='auto')
	# 			plt.xlabel("Frequency [MHz]")
	# 			plt.ylabel("Time [ms]")
	# 			plt.title("Input Trace")
	# 			plt.show()

