import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sw = pd.read_hdf("../data_local/omni/sheath_sw_data.h5", key="2017")

omni_data = pd.read_hdf("../data_local/omni/sw_data.h5", key="2017")
print(sw, omni_data)

start_index = np.argmin(np.abs(sw.index[0] - omni_data.index))
end_index = np.argmin(np.abs(sw.index[-1] - omni_data.index))
omni_data = omni_data.iloc[start_index:end_index]
omni_data["vx"] = np.abs(omni_data["vx"])

sw.reset_index(inplace=True, drop=False)
omni_data.reset_index(inplace=True, drop=False)


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.ravel()

out_feature_names = [
    "Time",
    "bx",
    "by",
    "bz",
    "vx",
    "vy",
    "vz",
    "density",
    "psw",
    "temperature",
    "xgse",
    "ygse",
    "zgse",
    "clock_angle",
]

for i, ai in enumerate([1, 2, 3, 4, 7, 9]):
    axes[i].plot(
        omni_data["index"], omni_data.iloc[:, ai], c="black", label="Actual" * (not i)
    )
    axes[i].plot(sw["Time"], sw.iloc[:, ai], c="red", label="Predicted" * (not i))
    axes[i].tick_params(axis="x", labelrotation=45)
    axes[i].set_title(out_feature_names[ai])
plt.tight_layout()
fig.legend()
plt.savefig("sw_timeseries.png")
