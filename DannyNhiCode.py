# %%
import phoebe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import TimeSeries

logger = phoebe.logger()
b = phoebe.default_binary(contact_binary=True)

# %%
# Download csv from github, read into pandas
url = 'https://raw.githubusercontent.com/WenhanGuo/contact-binaries/master/ts_3nights.csv'
df = pd.read_csv(url)
df.set_index(pd.DatetimeIndex(df['time']), inplace=True)
del df['time']

# Convert to astropy TimeSeries
ts = TimeSeries.from_pandas(df)
MJD = ts['time'].mjd
MJD = MJD - MJD[0]   # set MJD start from 0 for t0 argument
MJD = MJD % 0.3439788   # fold time into delta time

print(MJD)
print(type(MJD[0]))


fluxes = 10**(-ts['diff_mag']/2.5 + 10)

# %%
orbphases = phoebe.linspace(0,1,101)
meshphases = phoebe.linspace(0,1,51)
b.add_dataset('lc', times=MJD, fluxes=fluxes, dataset='lc01')
b.add_dataset('orb', compute_phases=orbphases, dataset='orb01')
b.add_dataset('mesh', compute_phases=meshphases, dataset='mesh01', columns=['teffs'])

# %%
# print(phoebe.list_online_passbands())
b.set_value('passband', 'SDSS:g')
b.set_value_all('ld_mode', 'lookup')
b.set_value_all('ld_mode_bol', 'lookup')
b.set_value_all('atm', 'ck2004')

b.set_value('pblum_mode', 'dataset-scaled')
b.set_value_all('gravb_bol', 0.32)
b.set_value_all('irrad_frac_refl_bol', 0.5)

b['period@binary'] = 0.3439788   # period = 0.34 day
b['t0_supconj'] = 0.14
b['incl@binary'] = 89.6
b['Av'] = 0.179

b['teff@primary'] = 5742
b['teff@secondary'] = 5600

b.flip_constraint('mass@primary', solve_for='sma@binary')
b['mass@primary@component'] = 1.25
b['q'] = 0.110

b['requiv@primary'] = 1.37

print(b.run_checks())   # check if run_compute is possible
print(b)

# %%
b.run_compute(model='default')

# %%
# simple plotting
b.plot('lc01', x='phase', ylim=(0.4*10**10,0.8*10**10), s=0.008, legend=True, show=True, save='./cb_visu_obs/lc.png')   # plot lc data and forward model
b.plot('mesh01', phase=0, legend=True, fc='teffs', ec='None', fcmap='viridis', show=True)   # plot mesh w/ temp color @t0
# animations
b.plot(y={'orb':'ws'}, ylim={'lc':(0.4*10**10,0.8*10**10)}, size=0.008, fc={'mesh':'teffs'}, ec={'mesh':'None'}, 
        fcmap='viridis', animate=True, save='./cb_visu_obs/animations_sync.gif')   # sync animation for lc, orb, mesh
b.plot('orb01', y='ws', legend=True, animate=True, save='./cb_visu_obs/orb2d.gif')   # animate face-on 2d orbit
b.plot('orb01', projection='3d', legend=True, animate=True, save='./cb_visu_obs/orb3d.gif')   # animate 3d orbit
b.plot('mesh01', fc='teffs', ec='None', fcmap='viridis', legend=True, animate=True, save='./cb_visu_obs/mesh.gif')   # animate mesh

# %%
# b.add_solver('estimator.lc_periodogram')
# b.run_solver(kind='lc_periodogram', lc_datasets='lc01')

# %%
# start of inverse problem: add and run KNN estimator
b.add_solver('estimator.ebai', ebai_method='knn', solver='ebai_knn', overwrite=True)
b.run_solver('ebai_knn', solution='ebai_knn_sol', phase_bin=False)
print(b.adopt_solution('ebai_knn_sol', trial_run=True))   # see proposed KNN solution params before adopting

# %%
b.flip_constraint('teffratio', solve_for='teff@secondary')

# if adopt all proposed params, uncomment below:
b.flip_constraint('pot@contact_envelope', solve_for='requiv@primary')
print(b.adopt_solution('ebai_knn_sol'))

# if not adopting q, uncomment below:
# print(b.adopt_solution('ebai_knn_sol', adopt_parameters=['t0_supconj', 'teffratio', 'incl']))

# %%
b.run_compute(model='ebai_knn_model', overwrite=True)

# %%
b.plot('lc01', x='phase', ylim=(0.4*10**10,0.8*10**10), ls='-', s=0.008, legend=True, show=True, save='./cb_visu_obs/lc_inverse_obs.png')
b.plot('mesh01', fc='teffs', ec='None', fcmap='viridis', animate=True, save='./cb_visu_obs/mesh_inverse_obs.gif')
b.plot(y={'orb':'ws'}, ylim={'lc':(0.4*10**10,0.8*10**10)}, size=0.008, fc={'mesh':'teffs'}, ec={'mesh':'None'}, 
        fcmap='viridis', animate=True, save='./cb_visu_obs/animations_sync_inverse_obs.gif')
b.plot('orb01', y='ws', legend=True, animate=True, save='./cb_visu_obs/orb2d_inverse_obs.gif')
b.plot('orb01', projection='3d', legend=True, animate=True, save='./cb_visu_obs/orb3d_inverse_obs.gif')
b.plot('mesh01', fc='teffs', ec='None', fcmap='viridis', animate=True, save='./cb_visu_obs/mesh_inverse_obs.gif')

# %%
b.add_solver('optimizer.nelder_mead', 
            fit_parameters=['teffratio', 'incl@binary', 'q', 'per0'], solver='nm_solver')
b.run_solver('nm_solver', maxiter=10000, solution='nm_sol')
print(b.adopt_solution('nm_sol', trial_run=True))

# %%
print(b.adopt_solution('nm_sol'))
b.run_compute(compute='fastcompute', model='after_nm')

# %%
b.add_solver('sampler.emcee', solver='emcee_solver')
b.set_value('compute', solver='emcee_solver', value='fastcompute')
b.set_value('pblum_mode', 'dataset-coupled')

b.add_distribution({'t0_supconj': phoebe.gaussian_around(0.01),
                    'teffratio@binary': phoebe.gaussian_around(0.1),
                    'incl@binary': phoebe.gaussian_around(5),
                    'fillout_factor@contact_envelope': phoebe.gaussian_around(0.5),
                    'q@primary': phoebe.gaussian_around(0.5),
                    'pblum@primary': phoebe.gaussian_around(0.2),
                    'sigmas_lnf@lc01': phoebe.uniform(-1e9, -1e4),
                   }, distribution='ball_around_guess')
b.run_compute(compute='fastcompute', sample_from='ball_around_guess',
                sample_num=10, model='init_from_model')
b.plot('lc01', x='phase', ls='-', model='init_from_model', show=True)

# %%
b['init_from'] = 'ball_around_guess'
b.set_value('nwalkers', solver='emcee_solver', value=14)
b.set_value('niters', solver='emcee_solver', value=500)

b.run_solver('emcee_solver', solution='emcee_solution')
print(b.adopt_solution(solution='emcee_solution', distribution='emcee_posteriors'))
b.plot_distribution_collection(distribution='emcee_posteriors', show=True)
b.uncertainties_from_distribution_collection(distribution='emcee_posteriors', sigma=3, tex=False)

# %%
b.plot(solution='emcee_solution', style='lnprobability',
            burnin=100, thin=1, lnprob_cutoff=3600,
            show=True)

# We can fix the following if we know which values result in a nice chain
b.set_value('burnin', 100)
b.set_value('thin', 1)
b.set_value('lnprob_cutoff', 3600)

# %%
# Show corner plot
b.plot(solution='emcee_solution', style='corner', show=True)

# %%
# Show corner plot with failed and rejected samples
b.plot(solution='emcee_solution', style='failed', show=True)

# %%
# Show history of each sampled parameter for all walkers
b.plot(solution='emcee_solution', style='trace', show=True)

# %%
b.run_compute(compute='fastcompute', sample_from='emcee_solution',
                sample_num=20, model='emcee_sol_model')
b.plot('lc01', x='phase', ls='-', model='emcee_sol_model', show=True)