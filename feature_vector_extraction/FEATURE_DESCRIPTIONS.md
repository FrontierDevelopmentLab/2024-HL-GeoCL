## Feature Vectors for DAGGER

The script generates feature vectors from space weather data to be used in the DAGGER system. Each feature vector consists of various parameters derived from the original dataset and additional computed metrics. Here is a list of the features included in the feature vectors along with descriptions of their roles and computations:

- **Time**: Timestamp of the data record.
- **bx (`bx_gsm`)**: The magnetic field component in the GSM X-direction.
- **by (`by_gsm`)**: The magnetic field component in the GSM Y-direction.
- **bz (`bz_gsm`)**: The magnetic field component in the GSM Z-direction.
- **bt**: Total magnetic field strength.
- **v (`proton_speed`)**: Speed of solar wind protons.
- **n (`proton_density`)**: Density of solar wind protons.
- **t (`proton_temperature`)**: Temperature of solar wind protons.
- **dipole_tilt**: The angle of the Earth's magnetic dipole axis relative to its rotational axis, which varies with time.
- **f107**: The F10.7 index, a measure of solar radio flux at 10.7 cm wavelength, a proxy for solar activity.
- **kp**: The Kp index, a global geomagnetic activity index based on 3-hour measurements.
- **hp30**: 30-minute averaged planetary magnetic field H component.
- **ap30**: 30-minute averaged planetary geomagnetic disturbance index.
- **clock_angle**: Calculated as the arctangent of `by` over `bz`, converted from radians to degrees. It indicates the orientation of the interplanetary magnetic field.
- **sqrt_f107**: Square root of the F10.7 index.
- **derived_1**: Product of total magnetic field strength and the cosine of the clock angle.
- **derived_2**: Product of solar wind speed and the cosine of the clock angle.
- **derived_3**: Product of dipole tilt and the cosine of the clock angle.
- **derived_4**: Product of the square root of F10.7 index and the cosine of the clock angle.
- **derived_5**: Product of total magnetic field strength and the sine of the clock angle.
- **derived_6**: Product of solar wind speed and the sine of the clock angle.
- **derived_7**: Product of dipole tilt and the sine of the clock angle.
- **derived_8**: Product of the square root of F10.7 index and the sine of the clock angle.
- **derived_9**: Product of total magnetic field strength and the cosine of twice the clock angle.
- **derived_10**: Product of solar wind speed and the cosine of twice the clock angle.
- **derived_11**: Product of total magnetic field strength and the sine of twice the clock angle.
- **derived_12**: Product of solar wind speed and the sine of twice the clock angle.
- **derived_13**: Square of the sine of the clock angle.
- **p**: Dynamic pressure of the solar wind calculated as `2*1e-6 * n * v^2`, providing an estimate of the momentum transfer from solar wind to the magnetosphere.
- **e**: Electric field component derived as `-v * bz * 1e-3`, representing the interaction of the solar wind with Earth's magnetic field.

These features encapsulate various aspects of solar and geomagnetic parameters, aiding in the analysis of space weather impacts and dynamics. The derived features especially contribute to understanding the complex interactions within the Earth's magnetosphere.
