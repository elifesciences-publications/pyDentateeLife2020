# pyDentateeLife2020

This is a legacy repository dedicated to the reproducibility of our findings from Braganza et al. 2020 that were generated with pyDentate. If you want to use or build on pyDentate it is highly recommended to use the [active pyDentate repository](https://github.com/danielmk/pyDentate).

# Running pyDentateeLife2020 legacy version

Follow these steps to run this legacy version of pyDentate:
<ol>
<li>Install <a href="https://www.anaconda.com/distribution">Anaconda</a></li>
<li>Install <a href="https://www.neuron.yale.edu/neuron">NEURON</a>
  <p>There are many ways to install NEURON. I prefer the <a href="https://anaconda.org/conda-forge/neuron">conda-forge</a> distribution<blockquote>
        <p>conda install -c conda-forge/label/cf201901 neuron</p>
    </blockquote></p>
</li>
<li>Install elephant
  <p><blockquote>pip install elephant</blockquote></li></p>
<li><a href="https://www.neuron.yale.edu/neuron/download/compile_mswin">Compile the NEURON mechanisms</a> in /mechs_7-6</li>
<li>Download the pyDentateeLife2020 repository and unpack</li>
<li>Open paradigm_pattern_separation.py and add the path to your compiled mechanisms to dll_files variable</li>
<li>Run paradigm_pattern_separation.py</li>
</ol>
If you encounter problems with running pyDentate or have questions feel free to contact me (muellerkomorowska@gmail.com or https://twitter.com/scidanm).

# pyDentate structure

paradigm_ files are scripts that create one or more networks, run them and save the results. net_ files contain the network classes that are used by scripts to create the networks. Other files define for example the input pattern generator (burst_generator_inhomogeneous_poisson.py) or the single cells specifications (granulecell.py with granuelcellparams.txt). ouropy provides a backend for the networks by defining the logic to create and connect networks. I custom made ouropy to move away from NEURONS Section logic to a higher-level logic of cells, populations, connections and finally networks. Networks in net_ files inherit from ouropys GenNetwork class. For more details on pyDentate make sure to look at the [active pyDentate repository](https://github.com/danielmk/pyDentate).

# Author

Daniel Müller-Komorowska - [Institute of Experimental Epileptology and Cognition Research](https://eecr-bonn.de/)
