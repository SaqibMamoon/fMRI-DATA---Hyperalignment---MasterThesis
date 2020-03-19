Andrea Bertana - 2014

This is the pipeline that I used for my Master thesis and the internshipt at FBK.
Details and final report are in ReportInternship.pdf and a poster presented at 
CAOS 2014 is poster_caos_last.pdf

The below introduction is extracted from the report. 

INTRODUCTION

During the last years, extracting inter individual correspondence has been one of the most
debated topics in current FMRI research. It is not difficult to understand in fact that having a
direct match between different participants brains give the possibility of addressing new scientific
questions by analyzing between participants differences and similarities.

This problem, has been firstly addressed by using anatomical alignment algorithms. With these
strategies each individual brain is warped to a common reference template (Talaraich, MNI) in
order to establish spatial correspondence between brains. Therefore, after brain normalization, a
point in the common space identified by its x, y, z coordinates is assumed to refer to a similar
region in any brain normalized according to the same procedure.

However, brain normalization has several problems. First of all, many algorithms fails on aligning
specific part of the brains making really hard to have perfect anatomical matching. Second, it
introduce a challenging fundamental problem of neuroscience that goes beyond pure
transformation issues since it includes the question about the consistency of structuralfunctional
relationships. Since now infact, many neuroimaging studies have demonstrated that on a macroanatomical
scale a functional correspondence between different participants exists for various brain regions. However, 
this macroanatomicalcoherence seems not to be respected on a smaller scale as often a multisubjects 
study using anatomicalalignment fails on finding those microdistinctionsthat help people to properly 
represent theoutside environment and discriminate across millions of complex visual stimuli.

In this context, functional hyperalignment (Haxby, 2011) turned out to be an effective multivariate
method to functionally align different participants’ brain using brain responses measured with
fMRI instead of spatial anatomical landmarks. Hyperaligment thus completely abstracts from
spatial information and uses a geometrical transformation, procrustean transform, over useful
functional data to build an high dimensional common space that minimize the overall Euclidean
distance between these sets of brain responses. Once the data are projected to this
commonspace, correspondence between brains is established and the validity of such model
can be tested through a between subject classification analysis (BSC). In a BSC the response
vector of one subject are classified based on predictions computed over others subjects
responses. This procedure allow to built a general valid model that can be valid across subjects
and thus that can generalize on response tuning function that are common across brains.
However, because of his novelty, several aspects of this method are still underinvestigated.
Here, as a first step in our investigation, we propose to study the sensibility of such methods
under three different standard preprocessing choices: motion correction, detrending and
smoothing. Furthermore, we compared BSC performance for brain response vector that had
been transformed in this commonspace to BSC performance of data that were anatomically
aligned and to withinsubject classification (WSC), in which response vectors for subject were
