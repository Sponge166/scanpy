import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scan_wrapper import scan_wrapper
from sigmf.sigmffile import SigMFFile, fromfile
from datetime import timedelta, datetime

if __name__ == "__main__":
	#characterize
	d = pd.read_csv("testdata/testfile1/test_file.csv", header=None)
	d = d.to_numpy()
	Omega, pred = scan_wrapper(d)

	final_pred = np.zeros(pred[0].shape)
	for p in pred:
		final_pred += p

	# grab metadata for plot formatting
	meta = fromfile("rx.sigmf-meta")
	cap = meta.get_capture_info(0)
	start_time = datetime.fromisoformat(cap[SigMFFile.DATETIME_KEY])
	center_freq = cap[SigMFFile.FREQUENCY_KEY]
	sampling_rate = meta.get_global_field(SigMFFile.SAMPLE_RATE_KEY)

	n_rows, fft_size = d.shape
	row_duration = fft_size/sampling_rate # units = seconds
	td = timedelta(seconds=row_duration)
	extent = [center_freq - sampling_rate/2, center_freq + sampling_rate/2, start_time, start_time + n_rows*td]

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(121)
	ax.yaxis_date()
	# ax.yaxis.set_major_locator(mdates.SecondLocator())
	# ax.yaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
	plt.imshow(final_pred, cmap=plt.cm.spring, aspect='auto', extent=extent)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("Time [μs]")
	plt.title("Total Characterization")

	ax = fig.add_subplot(122)
	ax.yaxis_date()
	# ax.yaxis.set_major_locator(mdates.SecondLocator())
	# ax.yaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
	plt.imshow(d, cmap=plt.cm.spring, aspect='auto', extent=extent)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("Time [μs]")
	plt.title("Input Trace")
	plt.show()


