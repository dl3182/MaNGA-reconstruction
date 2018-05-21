
import time

from BaseRun import *



start_time = time.time()
plate = 7977

ifu = 701

rssfile = getrss(plate=plate,ifu=ifu)

print('kernel')
base=BaseReconstruct(rssfile=rssfile,dimage=0.75,nkernel=151)
# print('Shepard spectrum')
# spectrumShep=ReconstructShep(base=base,dimage=0.5)
print('G spectrum')
spectrumG=ReconstructG(base=base,dimage=0.75,ratio=30)

G_cube=write(rssfile, spectrumG, 'G2-'+str(plate)+'-'+str(ifu)+'-LOGCUBE')
# Shep_cube=write(rssfile, spectrumShep, 'Shepard-'+str(plate)+'-'+str(ifu)+'-LOGCUBE')


stop_time= time.time()

print("Time = %.2f"%(stop_time-start_time))
