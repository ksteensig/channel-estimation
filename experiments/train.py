import cbn as cbn
import cbn_recv as cbn_recv

import rbn as rbn
import rbn_cov as rbn_cov

training_size = 200000
validation_size = 0.1 # 10% of training size

# receiver antennas
N = 16

# bits
L = 16

# snr between 5 and 30 dB
snr = [5, 30]

learning_rate = 0.001

resolution = 180
"""
for k in [4, 8]:
    model_rbn = rbn_cov.train_model(N, k, L,
                                   snr = snr,
                                   training_size = training_size,
                                   validation_size=validation_size,
                                   learning_rate=learning_rate,
                                   sort=False)


for k in [4, 8]:
    model_rbn = rbn.train_model(N, k, L,
                                   snr = snr,
                                   training_size = training_size,
                                   validation_size=validation_size,
                                   learning_rate=learning_rate,
                                   sort=False)


for k in [4]:
    model_cbn = cbn_recv.train_model(N, k, L,
                                   snr = snr,
                                   resolution = resolution,
                                   training_size = training_size,
                                   validation_size=validation_size,
                                   learning_rate=learning_rate)

"""
for k in [8]:
    model_cbn = cbn.train_model(N, k, L,
                                   snr = snr,
                                   resolution = resolution,
                                   training_size = training_size,
                                   validation_size=validation_size,
                                   learning_rate=learning_rate)
