---
layout: default
---

**I have read the instructions and agree with the terms.** [Return to the application](https://nf1.hope4kids.io/).

The **OPTIMA: Optic Pathway Tumor Impact Monitoring with AI** is a free, open-source web-based application designed at [Children's National Hospital](https://www.childrensnational.org/) 
for the segmentation and analysis of optic pathway gliomas associated with neurofibromatosis type 1 (NF1-OPG)  in  T1 magnetic resonance imaging (MRI). Developed in Python, this web tool aims to provide precise quantitative analysis of pediatric brain MRI, to support clinical decision-making in diagnosis and prognosis.  

With its user-friendly interface, **OPTIMA** provides automated segmentation 
and volumetric measurements within minutes after uploading the t1 MRI sequences. 
This software provides **state-of-the-art performance** powered by our benchmarked 
segmentation model.  

# Usage

This software currently requires T1 MRI sequences: native pre-contrast T1-weighted (t1n) or 
contrast enhanced T1-weighted (t1c). The MRI sequences should be 
uploaded in NIfTI format (*i.e.*, **.nii.gz**). Before uploading, 
we strongly recommend performing **de-identification** to remove any protected 
health information, including **defacing** if necessary. 

<!-- **Pre-processing** in the Segmenter is under development. At this time, 
we expect users to follow the standardized ["BraTS Pipeline"](https://arxiv.org/pdf/2404.15009) 
pre-processing, which includes co-registration of four sequences, and resampling to isotropic 1 mm resolution.  
Public tools such as the Cancer Imaging Phenomics Toolkit ([CaPTk](https://github.com/CBICA/CaPTk)) 
and Federated Tumor Segmentation ([FeTS](https://fets-ai.github.io/Front-End/process_data)) 
toolkits can be used for this purpose.   -->

Once the MRI sequences are uploaded, simply click check the box and push the **Start Segmentation** button to generate segmentation and volumetric measurements in the **Status** box. The process typically takes around 1 minute. Afterward, you can visualize both the image and the segmentation in axial, coronal, and sagittal views using the interactive **Sliders**. Segmentation results can be downloaded in NIfTI format by clicking the **Download Segmentation File** button.  

For **demonstration** purposes, we provide sample cases at the bottom of the page. 
Select one and click the **Start Segmentation** button to see how the software works.  

# Source Code

The current version of the software is ![v1.0](https://img.shields.io/badge/v1.0-brightgreen) 
and the source code is publicly available on GitHub 
([code](https://github.com/Precision-Medical-Imaging-Group/HOPE-Segmenter-Kids)) 
under license [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

The software is developed and maintained by the [Precision Medical Imaging](https://research.childrensnational.org/labs/precision-medical) lab
at Children’s National Hospital in Washington, DC, USA.  

**Copyright Notification**  Copyright 2024 Children's National Medical Center and Universidad Polit&eacute;cnica de Madrid.

# Contributors

This web app was made possible due to the efforts made by

[Abhijeet Parida](https://www.linkedin.com/in/a-parida/), [Zhifan Jiang](https://www.linkedin.com/in/zhifan-jiang-19917531/), [Syed Muhammad Anwar](https://www.linkedin.com/in/syed-anwar-5473b81a/), [Marius George Linguraru](https://www.linkedin.com/in/mglinguraru/)

# Citations

If you use and/or refer to this software in your research, please cite the following papers: 

* D. Capell&aacute;n-Mart&iacute;n, Z. Jiang, A. Parida, X. Liu, V. Lam, H. Nisar, A. Tapp, S. Elsharkawi, M. J. Ledesma-Carbayo, S. M. Anwar, M. G. Linguraru, "Model Ensemble for Brain Tumor Segmentation in Magnetic Resonance Imaging,"
*Accepted in: S. Bakas, et al. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. MICCAI Workshop BrainLes 2023*, *arXiv:2409.08232 \[eess.IV\]* (2024), doi: [10.48550/arXiv.2409.08232](
https://doi.org/10.48550/arXiv.2409.08232).

* Z. Jiang, D. Capell&aacute;n-Mart&iacute;n, A. Parida, X. Liu, M. J. Ledesma-Carbayo, S. M. Anwar, M. G. Linguraru, 
"Enhancing Generalizability in Brain Tumor Segmentation: Model Ensemble with Adaptive Post-Processing," 
*2024 IEEE International Symposium on Biomedical Imaging (ISBI)*, Athens, Greece, 2024, pp. 1-4, 
doi: [10.1109/ISBI56570.2024.10635469](https://ieeexplore.ieee.org/document/10635469/authors#authors).


# Disclaimer

* This software is provided without any warranties or liabilities and is intended for research purposes only. 
It has not been reviewed or approved for clinical use by the Food and Drug Administration (FDA) or any other federal or state agency. 

* This software does not store any information uploaded to the platform. 
All uploaded and generated data are automatically deleted once the webpage is closed. 

# Contact

* New features are continually being developed. To stay informed about updates, 
report issues or provide feedback, please fill out the 
**online form** [here](https://forms.gle/kjMni6tpN1whA4RNA) or contact **Abhijeet Parida** [support@hope4kids.io](mailto:support@hope4kids.io) .   

* For more information, collaboration, or to receive the model in a Docker container 
for testing on large datasets locally, please contact 
**Prof. Marius George Linguraru** [contact@hope4kids.io](mailto:contact@hope4kids.io) with details on your intended use. 

**I have read the instructions and agree with the terms.** [Return to the application](https://nf1.hope4kids.io/).
