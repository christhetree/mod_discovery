<div align="center">
<h1>Modulation Discovery with Differentiable Digital Signal Processing</h1>
<p>
    <a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
    <a href="https://gudgud96.github.io/about/" target=”_blank”>Hao Hao Tan</a>, and
    <a href="https://www.eecs.qmul.ac.uk/~josh/" target=”_blank”>Joshua D. Reiss</a>
</p>

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/arXiv-2404.07970-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2404.07970&#41;)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://christhetree.github.io/mod_discovery/)
[![Plugins](https://img.shields.io/badge/neutone-Plugins-blue)](https://christhetree.github.io/mod_discovery/index.html#plugins)
[![License](https://img.shields.io/badge/License-MPL%202.0-orange)](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)
</div>

<h2>Abstract</h2>
<hr>
<p>
Modulations are a critical part of sound design and music production, enabling the creation of complex and evolving audio.
Modern synthesizers provide envelopes, low frequency oscillators, and more parameter automation tools that allow users to modulate the output with ease.
However, determining the modulation signals used to create a sound is difficult, and existing sound-matching / parameter estimation systems are often uninterpretable black boxes or predict high-dimensional framewise parameter values without considering the shape, structure, and routing of the underlying modulation curves.
We propose a neural sound-matching approach that leverages modulation extraction, constrained control signal parameterizations, and differentiable digital signal processing (DDSP) to discover the modulations present in a sound.
We demonstrate the effectiveness of our approach on highly modulated synthetic and real audio samples, its applicability to different DDSP synth architectures, and investigate the trade-off it incurs between interpretability and sound-matching accuracy.
We make our code and audio samples available and provide the trained DDSP synths in a VST plugin.
</p>

<h2>Citation</h2>
<hr>
Accepted to the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), Lake Tahoe, CA, USA, 12 - 15 October 2025.

<pre><code>@inproceedings{mitcheltree2025modulation,
    title={Modulation Discovery with Differentiable Digital Signal Processing},
    author={Christopher Mitcheltree and Hao Hao Tan and Joshua D. Reiss},
    booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
    year={2025}
}
</code></pre>

<h2>Instructions for Reproducibility</h2>
<hr>

Coming soon.

[//]: # ()
[//]: # (<ol>)

[//]: # (    <li>Clone this repository and open its directory.</li>)

[//]: # (    <li>Initialize and update the submodules &#40;<code>git submodule update --init --recursive</code>&#41;.</li>)

[//]: # (    <li>)

[//]: # (    Install the requirements using <br><code>conda env create --file=conda_env_cpu.yml</code> or <br>)

[//]: # (    <code>conda env create --file=conda_env.yml</code><br> for GPU acceleration.<br>)

[//]: # (    <code>requirements_pipchill.txt</code> and <code>requirements_all.txt</code> are also provided as references, but are not needed when using the <code>conda_env.yml</code> files.)

[//]: # (    </li>)

[//]: # (    <li>The source code can be explored in the <code>acid_ddsp/</code> directory.</li>)

[//]: # (    <li>All models from the paper can be found in the <code>models/</code> directory.</li>)

[//]: # (    <li>All eval results from the paper can be found in the <code>eval/</code> directory.</li>)

[//]: # (    <li>All <a href="https://neutone.ai" target=”_blank”>Neutone</a> files for running the models and the acid synth implementations as a VST in a DAW can be found in the <code>neutone/</code> directory.</li>)

[//]: # (    <li>Create an out directory &#40;<code>mkdir out</code>&#41;.</li>)

[//]: # (    <li>)

[//]: # (    All models can be evaluated by modifying and running <code>scripts/test.py</code>.<br>)

[//]: # (    Make sure your <code>PYTHONPATH</code> has been set correctly by running a command like<br>)

[//]: # (    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/acid_ddsp/</code>,<br>)

[//]: # (    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/torchlpc/</code>, and<br>)

[//]: # (    <code>export PYTHONPATH=$PYTHONPATH:BASE_DIR/fadtk/</code>.)

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (    CPU benchmark values can be obtained by running <code>scripts/benchmark.py</code>.<br>)

[//]: # (    These will vary depending on your computer.)

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (    &#40;Optional&#41; All models can be trained by modifying <code>configs/abstract_303/train.yml</code> and running <code>scripts/train.py</code>.<br>)

[//]: # (    Before training, <code>scripts/preprocess_data.py</code> should be run to create the dataset. )

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (    &#40;Optional&#41; Custom <a href="https://neutone.ai" target=”_blank”>Neutone</a> models can be exported by modifying and running <code>scripts/export_neutone_models.py</code> or <code>scripts/export_neutone_synth.py</code>.)

[//]: # (    </li>)

[//]: # (    <li>)

[//]: # (    The source code is currently not documented, but don't hesitate to open an issue if you have any questions or comments.)

[//]: # (    </li>)

[//]: # (</ol>)
