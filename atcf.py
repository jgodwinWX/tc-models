# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 07:22:27 2021

@author: jason
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import datetime as dt
import seaborn as sns

from scipy.stats import gaussian_kde

def latlonConverter(x):
    if 'N' in x:
        return float(x.strip('N')) / 10.0
    elif 'S' in x:
        return -float(x.strip('S')) / 10.0
    elif 'E' in x:
        return float(x.strip('E')) / 10.0
    elif 'W' in x:
        return -float(x.strip('W')) / 10.0
    else:
        raise Exception('Invalid lat/lon code %s' % x)

def get_cmap(n,name='tab20'):
    return plt.cm.get_cmap(name,n)

def density_estimation(m1,m2):
    X,Y = np.mgrid[min(m1):max(m2):100j,min(m2):max(m2):100j]
    positions = np.vstack([X.ravel(),Y.ravel()])
    values = np.vstack([m1,m2])
    if values.shape[1] == 1:
        return None,None,None
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T,X.shape)
    return X,Y,Z

# settings
systemname = 'Hurricane Ida'                # formal name (e.g. "Hurricane Wilma")
savedir = 'images'                          # output directory
systemid = '09L'                            # system ID number (e.g. "09L")
map_limit = True                            # limit the map extent? otherwise plot entire domain
map_extent = [-95, -85, 20, 35]             # [west,east,south,north]
kde_levs = 11                               # how many color levels to plot for KDE
kde_color = 'plasma'                        # colormap to use for KDE plots
nhc_intensity = False                       # plot official intensity? (broken at the moment)
draw_counties = True                        # draw county maps?
atcf_file = \
    'http://hurricanes.ral.ucar.edu/realtime/plots/northatlantic/2021/al092021/aal092021.dat'

# import the ATCF file
headers=['BASIN','CY#','INIT','TECH','MODEL','TAU','LAT','LON','VMAX','MSLP','LEV','RAD',\
         'WINDCODE','RAD1','RAD2','RAD3','RAD4','POUTER','ROUTER','RMW','GUST','EYE','SUBREGION',\
             'MAXSEA','FCSTR','DIR','SPEED','NAME','DEPTH','SEAS','SEACODE','SEA1','SEA2','SEA3',\
                 'SEA4','RMK']
    
atcf_df = pd.read_csv(atcf_file,error_bad_lines=False,names=headers)

# get the GEFS members
gefs = atcf_df[atcf_df['MODEL'].str.contains('AP') | atcf_df['MODEL'].str.contains('AEMN') | \
    atcf_df['MODEL'].str.contains('AVNO')]
# get the GEPS members
geps = atcf_df[(atcf_df['MODEL']=='  CMC') | (atcf_df['MODEL'].str.contains(' CP')) | \
               (atcf_df['MODEL'].str.contains(' CEMN'))]
# combine the GEFS/GEPS for the NAEFS
naefs = atcf_df[(atcf_df['MODEL'].str.contains(' AP')) | (atcf_df['MODEL'].str.contains(' CP'))]

# get the latest model cycle
latest = atcf_df[(atcf_df['INIT']==atcf_df['INIT'].max()) & \
                 (~atcf_df['MODEL'].str.contains(' AP')) & (~atcf_df['MODEL'].str.contains(' CP'))]
gefs_latest = gefs[gefs['INIT']==gefs['INIT'].max()]
geps_latest = geps[geps['INIT']==geps['INIT'].max()]
naefs_latest = naefs[naefs['INIT']==naefs['INIT'].max()]

# get the initialization time
gefs_init = dt.datetime.strptime(str(gefs_latest['INIT'].unique()[0]),'%Y%m%d%H')
gefs_init = dt.datetime.strftime(gefs_init,'%Y/%m/%d %H:00 UTC')
geps_init = dt.datetime.strptime(str(geps_latest['INIT'].unique()[0]),'%Y%m%d%H')
geps_init = dt.datetime.strftime(geps_init,'%Y/%m/%d %H:00 UTC')
naefs_init = dt.datetime.strptime(str(naefs_latest['INIT'].unique()[0]),'%Y%m%d%H')
naefs_init = dt.datetime.strftime(naefs_init,'%Y/%m/%d %H:00 UTC')
early_init = dt.datetime.strptime(str(latest['INIT'].unique()[0]),'%Y%m%d%H')
late_init = early_init + dt.timedelta(hours=-6)
early_init = dt.datetime.strftime(early_init,'%Y/%m/%d %H:00 UTC')
latestamp = int(dt.datetime.strftime(late_init,'%Y%m%d%H'))
late_init = dt.datetime.strftime(late_init,'%Y/%m/%d %H:00 UTC')

# get the late cycle guidance
late_df = atcf_df[(atcf_df['INIT']==latestamp) & (~atcf_df['MODEL'].isin(latest['MODEL']))]
late_df = late_df[~late_df['MODEL'].str.contains(' AP')]
late_df = late_df[~late_df['MODEL'].str.contains(' CP')]
late_df = late_df[~late_df['MODEL'].str.contains(' NP')]

# get the unique models
early_models = latest['MODEL'].unique()
gefs_members = gefs_latest['MODEL'].unique()
geps_members = geps_latest['MODEL'].unique()
naefs_members = naefs_latest['MODEL'].unique()
late_models = late_df['MODEL'].unique()

# convert the latitude and longitude columns to numeric values
latest['LAT'] = latest['LAT'].apply(latlonConverter)
latest['LON'] = latest['LON'].apply(latlonConverter)
gefs_latest['LAT'] = gefs_latest['LAT'].apply(latlonConverter)
gefs_latest['LON'] = gefs_latest['LON'].apply(latlonConverter)
naefs_latest['LAT'] = naefs_latest['LAT'].apply(latlonConverter)
naefs_latest['LON'] = naefs_latest['LON'].apply(latlonConverter)
late_df['LAT'] = late_df['LAT'].apply(latlonConverter)
late_df['LON'] = late_df['LON'].apply(latlonConverter)
geps_latest['LAT'] = geps_latest['LAT'].apply(latlonConverter)
geps_latest['LON'] = geps_latest['LON'].apply(latlonConverter)

# get KDEs
Xgefs,Ygefs,Zgefs = density_estimation(gefs_latest['LON'],gefs_latest['LAT'])
Xec,Yec,Zec = density_estimation(latest['LON'],latest['LAT'])
Xlc,Ylc,Zlc = density_estimation(late_df['LON'],late_df['LAT'])
Xgeps,Ygeps,Zgeps = density_estimation(geps_latest['LON'],geps_latest['LAT'])
Xnaefs,Ynaefs,Znaefs = density_estimation(naefs_latest['LON'],naefs_latest['LAT'])

# plot the GEFS members
fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
ax.patch.set_visible(False)
if map_limit:
    ax.set_extent(map_extent, ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAKES, alpha=0.5)
if draw_counties:
    reader = shpreader.Reader('gis/cb_2018_us_county_500k.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties,ccrs.PlateCarree())
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

# create colorbar for pressure
cmap = plt.cm.RdYlBu
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
bounds = np.linspace(900,1000,21)
norm = mpl.colors.BoundaryNorm(bounds,cmap.N)

# plot the kde for GEFS members
plt.contourf(Xgefs,Ygefs,Zgefs,levels=np.linspace(0.001,Zgefs.max(),kde_levs),alpha=0.75,\
             cmap=kde_color)

for a,b in enumerate(gefs_members):
    lons,lats = gefs_latest[gefs_latest['MODEL']==b]['LON'],\
        gefs_latest[gefs_latest['MODEL']==b]['LAT']
    if b == ' AEMN':
        size=100
        width=2
        col='cyan'
        name='GFS Ensemble Mean'
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree(),label=name)
    elif b == ' AVNO':
        size=100
        width=2
        col='yellow'
        name='Deterministic GFS'
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree(),label=name)
    else:
        size=10
        width=0.5
        col='black'
        name=b.strip(' ')
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree())
    plt.scatter(lons,lats,c=gefs_latest[gefs_latest['MODEL']==b]['MSLP'],cmap=cmap,\
        transform=ccrs.PlateCarree(),s=size,norm=norm)
plt.colorbar(label='Minimum Sea-Level Pressure (mb)',shrink=0.75,extend='both')
plt.legend(fontsize=16)
plt.title('GEFS Track Guidance for %s\nInitialized: %s' % (systemname,gefs_init),size=24)
plt.savefig('%s/%s_gefs.png' % (savedir,systemid),bbox_inches='tight')
plt.close('all')

# plot the GEPS members
fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
ax.patch.set_visible(False)
if map_limit:
    ax.set_extent(map_extent, ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAKES, alpha=0.5)
if draw_counties:
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

# plot the kde for GEFS members
plt.contourf(Xgeps,Ygeps,Zgeps,levels=np.linspace(0.001,Zgeps.max(),kde_levs),alpha=0.75,\
             cmap=kde_color)

for a,b in enumerate(geps_members):
    lons,lats = geps_latest[geps_latest['MODEL']==b]['LON'],\
        geps_latest[geps_latest['MODEL']==b]['LAT']
    if b == ' CEMN':
        size=100
        width=2
        col='cyan'
        name='GEPS Ensemble Mean'
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree(),label=name)
    elif b == '  CMC':
        size=100
        width=2
        col='yellow'
        name='Deterministic CMC/GDPS'
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree(),label=name)
    else:
        size=10
        width=0.5
        col='black'
        name=b.strip(' ')
        plt.plot(lons,lats,linewidth=width,color=col,transform=ccrs.PlateCarree())
    plt.scatter(lons,lats,c=geps_latest[geps_latest['MODEL']==b]['MSLP'],cmap=cmap,\
        transform=ccrs.PlateCarree(),s=size,norm=norm)
plt.colorbar(label='Minimum Sea-Level Pressure (mb)',shrink=0.75,extend='both')
plt.legend(fontsize=16)
plt.title('GEPS Track Guidance for %s\nInitialized: %s' % (systemname,geps_init),size=24)
plt.savefig('%s/%s_geps.png' % (savedir,systemid),bbox_inches='tight')
plt.close('all')

# plot the NAEFS guidance
fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
ax.patch.set_visible(False)
if map_limit:
    ax.set_extent(map_extent, ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAKES, alpha=0.5)
if draw_counties:
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

# plot the kde for early cycle guidance
plt.contourf(Xnaefs,Ynaefs,Znaefs,levels=np.linspace(0.001,Znaefs.max(),kde_levs),alpha=0.75,\
             cmap=kde_color)
    
for a,b in enumerate(naefs_members):
    lons,lats = naefs_latest[naefs_latest['MODEL']==b]['LON'],\
        naefs_latest[naefs_latest['MODEL']==b]['LAT']
    plt.plot(lons,lats,linewidth=0.5,color='black',transform=ccrs.PlateCarree())
    plt.scatter(lons,lats,c=naefs_latest[naefs_latest['MODEL']==b]['MSLP'],cmap=cmap,\
        transform=ccrs.PlateCarree(),s=10,norm=norm)
plt.colorbar(label='Minimum Sea-Level Pressure (mb)',shrink=0.75,extend='both')
plt.title('NAEFS Track Guidance for %s\nInitialized: %s' % (systemname,naefs_init),size=32)
plt.savefig('%s/%s_naefs.png' % (savedir,systemid),bbox_inches='tight')
plt.close('all')

# plot the early cycle guidance
fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
ax.patch.set_visible(False)
if map_limit:
    ax.set_extent(map_extent, ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAKES, alpha=0.5)
if draw_counties:
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

# plot the kde for early cycle guidance
plt.contourf(Xec,Yec,Zec,levels=np.linspace(0.001,Zec.max(),kde_levs),alpha=0.75,cmap=kde_color)
    
cmap = get_cmap(len(early_models))
for a,b in enumerate(early_models):
    lons,lats = latest[latest['MODEL']==b]['LON'],latest[latest['MODEL']==b]['LAT']
    plt.plot(lons,lats,linewidth=0.5,c=cmap(a),transform=ccrs.PlateCarree(),label=b)
    plt.scatter(lons,lats,c='black',transform=ccrs.PlateCarree(),s=10)
plt.legend(fontsize=16)
plt.title('Early-Cycle Track Guidance for %s\nInitialized: %s' % (systemname,early_init),size=32)
plt.savefig('%s/%s_early.png' % (savedir,systemid),bbox_inches='tight')
plt.close('all')

# plot the late cycle guidance
fig = plt.figure(figsize=(20,15))
ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
ax.patch.set_visible(False)
if map_limit:
    ax.set_extent(map_extent, ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)
ax.add_feature(cfeature.LAKES, alpha=0.5)
if draw_counties:
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')

# plot the kde for late cycle guidance
plt.contourf(Xlc,Ylc,Zlc,levels=np.linspace(0.001,Zlc.max(),kde_levs),alpha=0.75,cmap=kde_color)

cmap = get_cmap(len(late_models))
for a,b in enumerate(late_models):
    lons,lats = late_df[late_df['MODEL']==b]['LON'],late_df[late_df['MODEL']==b]['LAT']
    plt.plot(lons,lats,linewidth=0.5,c=cmap(a),transform=ccrs.PlateCarree(),label=b)
    plt.scatter(lons,lats,c='black',transform=ccrs.PlateCarree(),s=10)
plt.legend(fontsize=16)
plt.title('Late-Cycle Track Guidance for %s\nInitialized: %s' % (systemname,late_init),size=32)
plt.savefig('%s/%s_late.png' % (savedir,systemid),bbox_inches='tight')
plt.close('all')

# intensity plots
datasets = [naefs_latest,late_df,latest]
dataset_names = ['NAEFS','Late-Cycle','Early-Cycle']
official = atcf_df[atcf_df['MODEL']==' OFCI'][atcf_df['INIT']==atcf_df['INIT'].max()]
initialized = dt.datetime.strptime(str(official['INIT'].unique()[0]),'%Y%m%d%H')
official['VALID'] = [initialized + dt.timedelta(hours=x) for x in official['TAU']]
for ix,dataset in enumerate(datasets):
    # get the valid times
    initialized = dt.datetime.strptime(str(dataset['INIT'].unique()[0]),'%Y%m%d%H')
    dataset['VALID'] = [initialized + dt.timedelta(hours=x) for x in dataset['TAU']]
    
    # set up the colorbar to shade boxes by median ensemble value
    median_vals = dataset.groupby('TAU')['VMAX'].median()
    norm = plt.Normalize(0.0,150.0)
    colors = plt.cm.plasma(norm(median_vals))
    
    # create the boxplot
    sns.set_style('darkgrid')
    fig,ax = plt.subplots(figsize=(20,15))
    if nhc_intensity:
        fig = sns.pointplot(x='VALID',y='VMAX',data=official,color='black',linewidth=2,ax=ax)
    fig = sns.boxplot(x='VALID',y='VMAX',data=dataset,palette=colors,ax=ax)
    
    # tick labels
    ax.set_xticklabels(labels=dataset['VALID'].sort_values().dt.strftime('%d/%H').unique(),\
                       rotation=90,fontsize=16)
    ax.set(yticks=np.arange(0,151,5))
    ax.tick_params(axis='y',labelsize=16)
    
    # category lines
    fig.axhline(34,linestyle='-',color='orange')
    fig.text(0.25,36,'Tropical Storm',weight='bold',size=14,color='orange')
    fig.axhline(64,linestyle='-',color='red')
    fig.text(0.25,66,'Hurricane/Category 1',weight='bold',size=14,color='red')
    fig.axhline(83,linestyle='--',color='gray')
    fig.text(0.25,85,'Category 2',weight='bold',size=14,color='gray')
    fig.axhline(96,linestyle='-',color='magenta')
    fig.text(0.25,98,'Major Hurricane/Category 3',weight='bold',size=14,color='magenta')
    fig.axhline(113,linestyle='--',color='gray')
    fig.text(0.25,115,'Category 4',weight='bold',size=14,color='gray')
    fig.axhline(137,linestyle='--',color='gray')
    fig.text(0.25,139,'Category 5',weight='bold',size=14,color='gray')
    
    # axis labels and chart title
    fig.set_xlabel('Valid Time (UTC)',fontsize=20)
    fig.set_ylabel('Maximum Sustained Winds (kt)',fontsize=20)
    fig.set_title('%s Intensity Forecasts for %s\nInitialized at %s' % \
                  (dataset_names[ix],systemname,naefs_init),fontsize=32)
    
    # colorbar
    cb = fig.figure.colorbar(plt.cm.ScalarMappable(cmap='plasma',norm=norm),shrink=0.75)
    cb.ax.tick_params(labelsize=16)
    cb.set_ticks(np.arange(0,151,10))
    cb.set_label(label='Maximum Sustained Winds (kt) of median member', size=16)
        
    # save figure
    plt.savefig('%s/%s_vmax_%s.png' % (savedir,systemid,dataset_names[ix].lower()),\
                bbox_inches='tight')
    plt.close('all')

'''
# KDE at each valid time
Xgefs_t = [None] * gefs_latest['TAU'].nunique()
Ygefs_t = [None] * gefs_latest['TAU'].nunique()
Zgefs_t = [None] * gefs_latest['TAU'].nunique()
zmax = 0.0
for ix,i in enumerate(gefs_latest['TAU'].unique()):
    Xgefs_t[ix],Ygefs_t[ix],Zgefs_t[ix] = \
        density_estimation(gefs_latest['LON'][gefs_latest['TAU']==i],\
                           gefs_latest['LAT'][gefs_latest['TAU']==i])
    if Zgefs_t[ix] is not None and Zgefs_t[ix].max() > zmax:
        zmax = Zgefs_t[ix].max()
    
    
for ix,i in enumerate(gefs_latest['TAU'].unique()):
    if Zgefs_t[ix] is None:
        continue
    fig = plt.figure(figsize=(20,15))
    ax = plt.axes(projection=ccrs.PlateCarree(),frameon=False)
    ax.patch.set_visible(False)
    ax.set_extent([-100, -70, 10, 35], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    plt.contourf(Xgefs_t[ix],Ygefs_t[ix],Zgefs_t[ix],\
                 levels=np.linspace(0.001,zmax,101),alpha=0.75)
    plt.savefig('%s/kde/%s_gefs_kde_%03d.png' % (savedir,systemid,i),bbox_inches='tight')
    plt.close('all')
 '''