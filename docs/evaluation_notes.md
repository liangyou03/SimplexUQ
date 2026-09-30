# Evaluation notes

The protocol uses one response-free stratification map consistently for
calibration and test within a comparison. A map built from the complete frozen
prediction pool is a transductive benchmark preprocessing step, not the same
design as learning a map on an independent prospective reference sample.
Validity arguments require their stated independence/exchangeability conditions;
response-free construction alone does not establish all such conditions.

Historical result series are not interchangeable. Some legacy configurations
lack a recorded fixed-map flag. PBMC main and auxiliary series can use two versus
five populated strata; UTKFace FullCP and D6 exact references use smaller samples.
Do not interpret these cross-series differences as pure method effects.

Mean radius is meaningful in score units within a task, not as cross-task volume
or utility. The inverse-scale weighted method is a diagnostic rule, not a
density-ratio covariate-shift baseline. BatchMVP is an additional implementation
with finite-grid/stopping choices, not a newly proposed method or an unconditional
finite-sample guarantee inferred from mean empirical coverage.

Data and predictions must be obtained or rebuilt separately. Rebuilt predictors
can change the results. Never substitute generated arrays for frozen benchmark
arrays without recording the source and protocol change.

## Numerical warnings

The small synthetic smoke configuration can produce infinite Mondrian radii when
a stratum has insufficient calibration support. Aggregating infinite radii can
produce NaN standard deviations and NumPy warnings. Do not interpret such a row
as a finite-efficiency estimate. The local test run also emitted floating-point
warnings in some DGP matrix operations; passing the shape/unit tests does not
certify numerical stability for every environment or experiment configuration.
