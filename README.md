## Analyses Thrombosis Structure – v1.0

Thanks for your interest.

Blood coagulation is a vital process for humans and other species. Following an injury to a blood vessel, a cascade of molecular signals is transmitted, inhibiting and activating more than a dozen coagulation factors and resulting in the formation of a fibrin clot that ceases the bleeding. In this process, Antithrombin (AT), encoded by the SERPINC1 gene is a key player regulating the clotting activity and ensuring that it stops at the right time. In this sense, mutations to this factor often result in thrombosis - the excessive coagulation that leads to the potentially fatal formation of blood clots that obstruct veins. Although this process is well known, it is still unclear why even single residue substitutions to AT lead to drastically different phenotypes. 

We designed a machine learning framework for the classification of Thombose (Analyses Thrombosis Structure) and although our training data was limited, after careful optimization, Analyzes Thrombosis Structure was able to identify properties related to patterns that quantify the pathology of thrombosis.

Here you will find the datasets and the source code used in the manuscript “[Computational Analyses Reveal Fundamental Properties of the AT Structure Related to Thrombosis]( https://doi.org/10.1093/bioadv/vbac098)”, by 

[Tiago J S Lopes](https://scholar.google.com.br/citations?user=U_7gGdsAAAAJ&hl=pt-BR&oi=sra), [Ricardo A Rios](https://scholar.google.com.br/citations?user=esk3iMYAAAAJ&hl=pt-BR), [Tatiane N Rios](https://scholar.google.com.br/citations?user=TbGWmKIAAAAJ&hl=pt-BR), [Brenno M Alencar](https://scholar.google.com.br/citations?user=EDu4gfUAAAAJ&hl=pt-BR&oi=sra), [Marcos V Ferreira](https://scholar.google.com.br/citations?user=vW2mWMwAAAAJ&hl=pt-BR), [Eriko Morishita](javascript:;).

### Project structure

> - **/dataset** - contains the datasets to reproduce our findings and create the figures.
>
> - **/src** - contains the source code for the machine learning framework and for other analyses.
>
> - **/results** - you can find the pre-trained classification models in this folder.
>
>   

## Run 

To run machine traditional machine learning models:

```bash
cd src
python experiment.py
```

To run our model `synclass`:

```bash
cd src
python experiment_synclass.py
```

If you find any issues with the code, please contact us: tiago-jose@ncchd.go.jp, ricardoar@ufba.br, tatiane.nogueira@ufba.br.

On the behalf of all of the authors, we appreciate your interest in Hema-Class and hope it is useful to your research.