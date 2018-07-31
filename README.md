# MASC_py
MASC - Model of Attention in Superior Colliculus for generating eye-movements from priority maps

Hossein Adeli, July 2018 hossein.adelijelodar@gmail.com

Adeli, H., Vitu, F., & Zelinsky, G. J. (2017). A model of the superior colliculus predicts fixation locations during scene viewing and visual search. Journal of Neuroscience, 37(6), 1453-1467. http://www.jneurosci.org/content/37/6/1453

Start with MASC_demo.ipynb
This demo shows how to use MASC for generating a eye-movements from a priority (saliency) map using the MASC_core.py. MASC is a general model of eye-movement and can be used to generate eye-movements from different priority/saliency maps. You can modify this file for you purpose. The saliency map in this example is made using Itti-Koch method. 

MASC_demo_multi_fixations.ipynb
This demo shows how to generate multiple fixations from one priority map using a simple Inhibition Of Return (IOR) mechanism. 

MASC_py_for_learning.ipynb
If you are interested in different steps of processing in MASC, play around with this file. 


Refer to our JoN paper for general methods.
