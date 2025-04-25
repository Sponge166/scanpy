import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scan_wrapper import scan_wrapper
from sigmf.sigmffile import SigMFFile, fromfile
from datetime import timedelta, datetime
from matplotlib.ticker import FuncFormatter
import pylab as pl

if __name__ == "__main__":
	#characterize
	# d = pd.read_csv("testdata/ota/USRP_1000000.0_K1_06_27_2022_09_37_06_F1024.csv", header=None)
	d = pd.read_csv("testdata/testfile1/test_file.csv", header=None)
	d = d.to_numpy()
	# d = d[:2048]
	# print(d.shape)
	# gt1 = pd.read_csv("testdata/testfile1/gt1.csv", header=None)
	# gt1 = gt1.to_numpy()
	# gt2 = pd.read_csv("testdata/testfile1/gt2.csv", header=None)
	# gt2 = gt2.to_numpy()
	Omega, pred = scan_wrapper(d)

	final_pred = np.zeros(pred[0].shape)
	for p in pred:
		final_pred += p

	# grab metadata for plot formatting
	# meta = fromfile("testdata/ota/rx.sigmf-meta")
	meta = fromfile("rx.sigmf-meta")
	cap = meta.get_capture_info(0)
	start_time = datetime.fromisoformat(cap[SigMFFile.DATETIME_KEY])
	center_freq = cap[SigMFFile.FREQUENCY_KEY]
	sampling_rate = meta.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
	sensor_model = meta.get_global_field("ntia-sensor:sensor", dict()).get("sensor_spec",dict()).get("model")
	antenna_model = meta.get_global_field("ntia-sensor:sensor", dict()).get("antenna",dict()).get("antenna_spec",dict()).get("model")

	n_rows, fft_size = d.shape
	row_duration = fft_size/sampling_rate # units = seconds
	td = timedelta(seconds=row_duration)
	end_time = start_time + n_rows*td
	extent = [center_freq - sampling_rate/2, center_freq + sampling_rate/2, start_time, end_time]

	# plot

	@FuncFormatter
	def my_formatter(x, pos):
		return pl.num2date(x).strftime("%M:%S.%f").removesuffix("00000").removesuffix("0000").removesuffix("000")

	fig = plt.figure()
	ax = fig.add_subplot(121)
	ax.yaxis_date()
	ax.yaxis.set_major_formatter(my_formatter)
	plt.imshow(final_pred, cmap=plt.cm.spring, aspect='auto', extent=extent)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("Time [μs]")
	plt.title("Total Characterization")

	ax = fig.add_subplot(122)
	ax.yaxis_date()
	ax.yaxis.set_major_formatter(my_formatter)
	plt.imshow(d, cmap=plt.cm.spring, aspect='auto', extent=extent)
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("Time [μs]")
	title_str = "Input Trace"
	if sensor_model:
		title_str += f" (Sensor: {sensor_model}"
	if antenna_model:
		title_str += f", Antenna: {antenna_model})"
	else:
		title_str += ")"
	plt.title(title_str)

	plt.show()


