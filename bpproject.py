import torch
import numpy as np

from bpnetlite.io import extract_loci
from bpnetlite.io import PeakGenerator
from bpnetlite import BPNet
#from bpnetlite.io import extract_peaks

peaks = 'chip-nexus/Oct4/idr-optimal-set.narrowPeak.gz' # A set of loci to train on.
seqs = 'mm10_no_alt_analysis_set_ENCODE.fasta' # A set of sequences to train on
signals = ['chip-nexus/Oct4/counts.pos.bw', 'chip-nexus/Oct4/counts.neg.bw'] # A set of bigwigs
controls = ['chip-nexus/patchcap/counts.pos.bw', 'chip-nexus/patchcap/counts.neg.bw'] # A set of bigwigs
training_chroms = ['chr{}'.format(i) for i in range(1, 17)]

training_data = PeakGenerator(peaks, seqs, signals, controls, chroms=training_chroms)
valid_chroms = ['chr{}'.format(i) for i in range(18, 23)]

X_valid, y_valid, X_ctl_valid = extract_loci(peaks, seqs, signals, controls, chroms=valid_chroms, max_jitter=0)
#print(X_valid.size(),y_valid.size(),X_ctl_valid.size())

model = BPNet(n_outputs=2, n_control_tracks=2, trimming=(2114 - 1000) // 2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.fit(training_data, optimizer, X_valid=X_valid, X_ctl_valid=X_ctl_valid, y_valid=y_valid)
#model.fit(training_data, optimizer, X_valid=X_valid, X_ctl_valid=X_ctl_valid, y_valid=y_valid,max_epochs=10, batch_size=64, validation_iter=100)
state_dict = model.state_dict()
torch.save(state_dict, 'model_Oct4.pth')
model=model.cuda().float()
X=X_valid.cuda().float()
X_ctl=X_ctl_valid.cuda().float()
[y_profiles,y_counts]=model.predict(X, X_ctl)
y_profiles.numpy()
y_counts.numpy()
y_profiles_2d = np.reshape(y_profiles, (y_profiles.shape[0], y_profiles.shape[1]*y_profiles.shape[2]))
np.savetxt("y_profiles_Oct4.txt", y_profiles_2d, fmt="%f", delimiter=",")
np.savetxt("y_counts_Oct4.txt", y_counts)