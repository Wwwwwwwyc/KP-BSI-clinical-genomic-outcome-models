# KP-BSI clinical-genomic outcome models

Methodology code and model parameters for predicting 30-day mortality and liver abscess or metastatic infection in *Klebsiella pneumoniae* bloodstream infection.

## Contents

- **Model development:** outcome-specific clinical feature selection, population-structure-adjusted GWAS, nested clinical-genomic models and compact-model development.
- **Association analysis:** adjusted clinical-genomic associations and individual-marker comparisons.
- **Model evaluation:** discrimination, calibration, classification performance, confidence intervals and decision curves.
- **External validation:** genomic predictor reconstruction and application of the compact model to an independent cohort.
- **Model parameters:** the nine-feature model's coefficients, imputation medians, scaling parameters, Platt calibration and classification threshold in `parameters/compact_model.json`.

## Methods

Training and feature selection use 2013–2021 episodes. Model selection, calibration and threshold selection use 2022–2023 episodes; temporal testing uses 2024–2025 episodes. Nested models combine clinical predictors, antimicrobial susceptibility testing, five hypervirulence-associated markers and selected pangenome features.

The compact model includes hepatitis, *iroB*, *fimA*, continuous Pitt score, cancer, *bla*<sub>KPC</sub>, organ transplantation, *yhdJ* and diabetes. Feature and penalty selection use the training and validation cohorts. The final compact model, including imputation and standardisation, is fitted on the combined 2013–2023 development cohort. Platt calibration and the classification threshold are estimated from validation predictions made by the training-set model and retained for model application.

AUROC confidence intervals use percentile bootstrap resampling. Performance differences are calculated directly from the evaluation sample, with paired percentile bootstrap confidence intervals. Confidence intervals for sensitivity, specificity, PPV and NPV at locked thresholds use the Wilson method.

## External validation

`evaluate_external.py` applies the compact model to external predictors and reports probabilities, AUROC with a stratified percentile 95% confidence interval from 2,000 bootstrap resamples, and sensitivity and specificity at the validation-selected threshold. The model is applied without external refitting or recalibration.

`evaluate_gan_cohort.py` reconstructs genomic predictors for the independent Chinese cohort reported by Gan et al. (doi:10.1128/spectrum.02646-21). This comparison includes 124 pyogenic-liver-abscess-associated and 70 pneumonia-associated isolates. TBLASTN against 34 protein alleles identifies *iroB*, and BLASTN against the selected gene-family references identifies *fimA* and *yhdJ*. Matches require ≥90% identity and ≥80% reference query coverage in a single alignment, with E-value ≤10⁻⁵. Published KPC annotations provide the *bla*<sub>KPC</sub> predictor with BLASTN sequence verification. Unavailable clinical predictors receive the model's imputation values.

## Usage

Install the Python packages with `pip install -r requirements.txt`, or create the Conda environment with `conda env create -f environment.yml`. GWAS uses pyseer and FastTree; external genomic reconstruction uses BLAST+.

- `scripts/run_methodology_pipeline.py`: combined model-development and evaluation analyses.
- `03_models/clinical_genomic_models/scripts/`: individual model-development analyses.
- `04_analysis/association/scripts/`: association analyses.
- `04_analysis/supplementary_statistics/scripts/`: performance and interval estimation.
- `04_analysis/external_validation/scripts/`: external model application and evaluation.

Run an individual script without arguments to display its input and output options.

## Citation and licence

Please cite the associated manuscript when using these methods. Code and model parameters are distributed under the MIT License; see `LICENSE`.
