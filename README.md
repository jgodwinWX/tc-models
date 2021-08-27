# tc-models
 Plots tropical cyclone guidance (track and intensity) from ATCF. ATCF files can be found from the
 NCAR Tropical Cyclone Guidance Project: http://hurricanes.ral.ucar.edu/realtime/current/ . Sample
 plots are available in the images/ directory. The images/archived directory contains old plots
 that are no longer being generated. The images/kde directory contains individual timestep KDE
 filled-contour plots. These are not being generated currently, but the plot to create them is
 commented out at the bottom of atcf.py.
 
 Current capabilities:
 -Track Guidance:
     - GEFS, GEPS, NAEFS (GEFS+GEPS), early-cycle, and late-cycle tracks (line plots).
     - Kernel density estimation (KDE) for each, plotted as a filled contour.
 -Intensity Guidance:
     - Box and whisker plots for NAEFS, early-cycle, and late-cycle. Boxes are colored based on
     the median member.
     
Ongoing issues:
1. There is an option to plot the NHC intensity guidance on top of the box-and-whisker plots, but
this is currently broken. I am having difficult getting the box-and-whiskers (guidance) and point
plot (NHC official) to "share" the x-axis. Suggestions welcomed!
2. There are numerous SettingWithCopyWarnings that need to be resolved, but right now, do not 
appear to cause any major problems.

Future work:
1. Neighborhood probabilities: plot probabiltiies of ensemble members within 200 km. This is quite
tricky and involves working with meshgrids, something with which I have no experience. Suggestions
welcomed!

Dependencies:
- Cartopy (version=0.18.0)
- matplotlib (version=3.2.2)
- numpy (version=1.18.5)
- pandas (version=1.0.5)
- seaborn (version=0.10.1)

* The versions listed are what I am running. I cannot say how an earlier/later version would effect
this, but a significantly earlier version might cause things to break, as I experienced by running
this code on two different machines. This code was created in an Anaconda environment on Windows 
10, but did run on Ubuntu Linux as well.

Any feedback, questions, comments, tips, and suggestions are welcomed! You can reach me via email
or by private message on Twitter (@jgodwinWX).