= Reproducible research : Python implementation of SparseHebbianLearning =

||<tablestyle="width: 35%; float: right; margin-left:20px; margin-right:20px; border-style: 0px;  font-size: 8pt;"> [[SparseHebbianLearning|{{attachment:SparseHebbianLearning/ssc.gif|Animation of the formation of RFs during aSSC learning|width=100%,align="right"}}]] <<BR>> ''Animation of the formation of RFs during aSSC learning.'' ||

 /!\ Read before using! This piece of code is distributed under the terms of the GNU General Public License (GPL), check http://www.gnu.org/copyleft/gpl.html if you have not red the term of the license yet. 

 *  (!)  tl;dr : [[attachment:assc_python.zip|Download the code|&do=get]]. Or directly from the command-line, do {{{

wget https://github.com/meduz/SHL_scripts/archive/master.zip
unzip master.zip -d assc_python
cd assc_python
python learn.py
}}}

== Object ==

 * This is a collection of python scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the !SparseNet algorithm  from Bruno Olshausen.

 * this has been published as Perrinet, Neural Computation (2010) (see  http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl ): {{{#!bibtex
@article{Perrinet10shl,
    Author = {Perrinet, Laurent U.},
    Doi = {10.1162/neco.2010.05-08-795},
    Journal = {Neural Computation},
    Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
    Month = {July},
    Number = {7},
    Title = {Role of homeostasis in learning sparse representations},
    Url = {http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl},
    Volume = {22},
    Year = {2010},
    Annote = {Posted Online March 17, 2010.},
}
}}}
 * all comments and bug corrections should be submitted to Laurent Perrinet at Laurent.Perrinet@gmail.com
 * find out updates on http://invibe.net/LaurentPerrinet/SparseHebbianLearning

== Get Ready! ==

 Be sure to have :
 * a computer (tested on Mac, Linux) with python + numpy (tested using the [[http://www.enthought.com/products/epd.php|Enthought Python Distribution]] version 7.1 on macosx and the standard python packages on ubuntu 11.04). You will not need any special toolbox, only {{{progress bar}}}.
 * grab the sources from the [[http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile&do=get&target=assc_python.zip|zip file]] or from [[https://github.com/meduz/SHL_scripts/archive/master.zip|github]]. 
 * These scripts should be platform independent, however, there is a heavy bias toward UN*X users when generating figures.

== Contents ==

 * {{{README.txt}}} : this file
 * {{{learn.py}}} : the scripts (see Contents.m  for a script pointing to the different experiments)
 * {{{sec.py}}} : the individual experiments
 * {{{IMAGES_*.mat}}} : the image files (if absent, they get automagically downloaded from [[http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile|this page]])


== Some useful code tidbits ==
 * get the code with CLI  {{{
wget https://github.com/meduz/SHL_scripts/archive/master.zip
}}}.
 * decompress  {{{
unzip master.zip -d assc_python
}}}
 * get to the code {{{
cd assc_python
}}}

 * run the main script {{{
python learn.py
}}}

 * remove SSC related files to start over {{{
rm -f IMAGES_*.mat.pdf *.hdf5
}}}

== Changelog ==

 * 1.0 : initial release, 27-Oct-2011

== TODO ==

 * integrate to [[http://mdp-toolkit.sourceforge.net/|MDP]], the Modular toolkit for Data Processing.

= Reproducible research : matlab(c) implementation of SparseHebbianLearning =

 * see also https://github.com/meduz/SHL_scripts

 . (!) [[attachment:SHL_scripts.zip|Download and run matlab code along with data (natural images) to reproduce figures|&do=get]]. 

 /!\ Read before using! This piece of code is distributed under the terms of the GNU General Public License (GPL), check http://www.gnu.org/copyleft/gpl.html if you have not red the term of the license yet. 

== Object ==

 * This is a collection of matlab (c) scripts to test learning strategies to efficiently code natural image patches.  This is here restricted  to the framework of the !SparseNet algorithm  from Bruno Olshausen.

 * this has been published as Perrinet, Neural Computation (2010) (see  http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl ): {{{#!bibtex
@article{Perrinet10shl,
    Author = {Perrinet, Laurent U.},
    Date-Added = {2007-06-25 15:49:50 +0200},
    Date-Modified = {2010-04-24 17:52:43 +0200},
    Doi = {10.1162/neco.2010.05-08-795},
    Journal = {Neural Computation},
    Keywords = {Neural population coding, Unsupervised learning, Statistics of natural images, Simple cell receptive fields, Sparse Hebbian Learning, Adaptive Matching Pursuit, Cooperative Homeostasis, Competition-Optimized Matching Pursuit},
    Month = {July},
    Number = {7},
    Title = {Role of homeostasis in learning sparse representations},
    Url = {http://invibe.net/LaurentPerrinet/Publications/Perrinet10shl},
    Volume = {22},
    Year = {2010},
    Annote = {Posted Online March 17, 2010.},
}
}}}
 * This includes a set of "experiments" to test the effect of various parameters (and eventually a 'good-night, computer' {{{Contents.m}}} script for running all experiments and generate all figures that are included in the "report"). I recommend using [[GnuLinuxUbuntu/UsingScreen|Gnu Screen]].

 * all comments and bug corrections should be submitted to Laurent Perrinet at Laurent.Perrinet@gmail.com
 * find out updates on http://invibe.net/LaurentPerrinet/SparseHebbianLearning

== Get Ready! ==

 Be sure to have :
 * a computer (tested on Mac, Linux, Irix, Windows2k) with Matlab (tested on R13 and R14, 2007, R2009a) or [[http://www.gnu.org/software/octave/|Octave]] (get  Octave > 3.0 to get {{{imwrite.m}}}). You will not need any special toolbox.
 * grab the sources from the [[http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile&do=get&target=SHL_scripts.zip|zip file]]. Then:
  * if needed (that is, if the code does break complaining it does not find the {{{cgf}}} function), compile the {{{cgf}}} routines used by B. Olshausen compiled for your platform (some compiled mex routines are included)
  * to generate PDFs, you have to get the {{{epstopdf}}} script (see {{{fig2pdf.m}}}) 
  * the source files {{{exportfig}}} and {{{ppm2fli}}} may be found in the src folder,
  * to generate the final report, you'll need a TeX distribution with the {{{pdflatex}}} program and the beamer package,
  * you will need to have a set of decorrelated images in your {{{./data}}} folder (its provided in the zip file, but you may make your own using the {{{src/spherize_images.m}}} script),
  * These scripts should be platform independent, however, there is a heavy bias toward UN*X users when generating figures (I haven't tried to generate the figures on windows systems). In particular, it is designed to generate figures in the background as PDF (on a headless cluster), and no window from MATLAB should pop up.

== Instructions for running the experiments / understanding the scripts ==

 * First, if you just want to experiment with the learning scheme using Competition-Optimized  Matching Pursuit, go to the scripts folder and run {{{experiment_simple.m}}}
 * Simply run one of the {{{experiment_*.m}}} files of your interest ---for example {{{experiment_stability_cgf.m}}} to test the role of parameters in the learning scheme with CGF--- (or the whole collection in {{{Contents.m}}}) and edit it to change the parameters of the experiments. This will create a set of pdf figures in a dated folder depending on your preferences (see {{{default.m}}})
 * the {{{Contents.m}}} script points to the different experiments. This produces a report using pdflatex: {{{pdflatex results.tex}}} (see [[http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile&do=get&target=results.pdf|results.pdf]]).
 *  Notation is kept from the !SparseNet package. Remember for the  variables : n=network ; e=experiment; s=stats
   
 * on a multicore machine, you may try something like: {{{
for i in {1..8}; do cd /Volumes/perrinet/sci/dyva/lup/Learning/SHL_scripts/code  && sleep 0.$(( RANDOM%1000 )) ; matlab -nodisplay < Contents20100322T151819.m & done
for i in {1..6}; do cd /master0/perrinet/sci/dyva/lup/Learning/SHL_scripts/code  && sleep 0.$(( RANDOM%1000 )) ; matlab -nodisplay < Contents20100322T151819.m & done
for i in {1..4}; do cd /data/work/perrinet/sci/dyva/Learning/SHL_scripts/code && sleep 0.$(( RANDOM%1000 )) ; octave  Contents20100322T151819.m & cexec 'cd /data/work/perrinet/sci/dyva/Learning/SHL_scripts/code && sleep 0.$(( RANDOM%100 )) ;  octave Contents20100322T151819.m' & done 
}}} 
== Contents ==


 * code : the scripts (see Contents.m  for a script pointing to the different experiments)
 * results : the individual experiments
 * data : a folder containing the image files (you van get them independently by downloading [[attachment:data.zip]])
 * src : some other package that may be of use


== Some useful code tidbits ==
 * get the code {{{
wget "http://invibe.net/LaurentPerrinet/SparseHebbianLearning/ReproducibleResearch?action=AttachFile&do=get&target=SHL_scripts.zip"
unzip ReproducibleResearch\?action\=AttachFile\&do\=get\&target\=SHL_scripts.zip 
}}}
 * get to the code {{{
cd SHL_scripts/code
}}}
 * begin a session [[UsingScreen|using GNU screen]]:{{{
screen}}} 
 * start multiple MATLAB sessions {{{
matlab -nodisplay < Contents20100322T151819.m
}}}
 * check latest mat files produced {{{
../results/20100322T151819/*.mat |tail -n30
}}}
 * check processes running {{{
top
}}}
 * transfer files to another computer {{{
rsync -av ../../SHL_scripts 10.164.2.49:~/Desktop
}}}
 * once finished, compile a report {{{
pdflatex results.tex
pdflatex results.tex
}}}

 * remove SSC related files to start over {{{
rm -f fig* *ssc* MP.mat L* stability_eta.mat stability_homeo.mat hist.mat MP_nonhomeo_lut.mat MP_nonhomeo.mat  MP_no_sym* perturb.mat stability_oc*.mat MP_icabench_decorr.mat MP_yelmo.mat 
}}}
== Changelog ==
 * 2.2 <- 2.1, 1-jan-2011
  * waited for official publication

 * 2.1 <- 2.0, 10-feb-2010
  * last code clean-up for publication

 * 2.0 <- 1.5, 10-dec-2009
  * lots of new figures for the reviewers
  * clean-up of the code for publication
  * a lot of bug fixes and speed improvements
  * made it Octave-compatible since I do not have Matlab anymore :-) 

 * 1.5 <- 1.4, 10-nov-2007
  * a lot of bug fixes and speed improvments
  * testing binary coding more extensively
  * added scripts to test alpha MP and the perturbation of a learning

 * 1.4 <- 1.3
  * bug fixes
  * better image format, script to make your own input data
  * more control experiments


 * 1.3 <- 1.2.1
  * GPL License
  * study adaptive gain (natural gradient) - implemented through switch_ng
  * fixed a lot of little bugs

 * 1.2.1 <- 1.2 <- 1.1, minor corrections, 05-Dec-2004
  * fixed some performance issues
  * added experiments
  * generates a report

 * 1.1 <- 1.0, 10-Nov-2004
  * clean-up / tried to simplify scripts (removed learning.m, allowed mp_fitS to process a whole batch, ...)
  * added experiments _fft.m, _ homeo.m, _symmetric.m, modified the others
  * may now load non-square images ( to load IMAGES_icabench )
  * better Windows reactions (duh!)

 * 1.0 : initial release, 10-Nov-2003
----
TagCode
