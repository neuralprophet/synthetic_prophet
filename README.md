# Synthetic Prophet
_Synthetic Data for Neural Prophet_

**Synthetic Prophet** is part of the Neural Prophet family. 
It seeks to generate synthetic time series for model testing. 
The library can generate regular and irregular time series. 
The architecture
allows the user to match different signals with different architectures allowing
a vast array of signals to be generated. The available signals and noise types are
listed below.
 ***Requirements***
- Python 3

#### Signal Types
* Harmonic functions(sin, cos or custom functions)
* Gaussian processes with different kernels
    * Constant
    * Squared exponential
    * Exponential
    * Rational quadratic
    * Linear
    * Matern
    * Periodic
* Pseudoperiodic signals
* Autoregressive(p) process
* Continuous autoregressive process (CAR)
* Nonlinear Autoregressive Moving Average model (NARMA)

#### Noise Types
* White noise
* Red noise

### Installation
To install the package via github,
```{bash}
git clone git@github.com:neuralprophet/synthetic_prophet.git
cd synthetic_prophet
python setup.py install
```

### Development
git clone <copied link from github>
cd synthetic_prophet
pip install -e ".[dev]"
syntheticprophet_dev_setup
git config pull.ff only 


### Using SyntheticProphet
```shell
$ python
```
The code snippet demonstrates creating a irregular sinusoidal signal with white noise.

```python
>> > import syntheticprophet as sp
>> >  # Initializing TimeSampler
>> > time_sampler = sp.TimeSampler(stop_time=20)
>> >  # Sampling irregular time samples
>> > irregular_time_samples = time_sampler._sample_irregular_time(num_points=500, keep_percentage=50)
>> >  # Initializing Sinusoidal signal
>> > sinusoid = sp.signals.Sinusoidal(frequency=0.25)
>> >  # Initializing Gaussian noise
>> > white_noise = sp.noise.GaussianNoise(std=0.3)
>> >  # Initializing TimeSeries class with the signal and noise objects
>> > syntheticseries = sp.SyntheticSeries(sinusoid, noise_generator=white_noise)
>> >  # Sampling using the irregular time samples
>> > samples, signals, errors = syntheticseries.sample(irregular_time_samples)
```

**Acknowledgements** 
This work gathers ideas and implementations from works such as:
- J. R. Maat, A. Malali, and P. Protopapas, “TimeSynth: A Multipurpose Library for Synthetic Time Series in Python,” 2017. [Online]. Available: http://github.com/TimeSynth/TimeSynth
- Fawaz, H.I. et al.: Data augmentation using synthetic data for time series classification with deep residual networks, http://arxiv.org/abs/1808.02455, (2018).
- SDV - The Synthetic Data Vault
- GANetano - Generative Adversarial Network to create synthetic time series
- Synthetic Time Series by blackarbsceo
- and others
Without them, we would not be here today.