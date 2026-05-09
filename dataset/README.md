# Dataset Directory

This directory is a local placeholder for prepared benchmark datasets. The
repository does not redistribute raw or converted dataset files.

## Required Format

Save each prepared dataset as a two-dimensional numeric NumPy array:

```text
dataset/<name>.npy
```

The last column must be the class label. All earlier columns are treated as
features by the loader.

```text
feature_1, feature_2, ..., feature_n, label
```

Large raw files and converted arrays are intentionally ignored by Git.

## Download Sources

The command-line examples use dataset names such as `covtype`, `electricity`,
`occupancy`, `shuttle`, and `kddcup99`. Download the raw data from the original
providers, preprocess it locally, and save the resulting array under the
matching filename.

Use the official dataset pages below. For MOA datasets, the MOA website is the
best human-readable index, while the individual archive files are hosted on
SourceForge.

| CLI name | Save as | Recommended source | Raw file or entry to use | Preparation notes |
| --- | --- | --- | --- | --- |
| `covtype` | `dataset/covtype.npy` | UCI Covertype: https://archive.ics.uci.edu/dataset/31/covertype; MOA dataset index: https://moa.cms.waikato.ac.nz/datasets/ | `covtype.data.gz` from UCI, or `covtypeNorm.arff.zip` from MOA/SourceForge | The UCI file already places `Cover_Type` in the final column. The MOA variant is normalized; record which version you use. |
| `electricity` | `dataset/electricity.npy` | MOA dataset index: https://moa.cms.waikato.ac.nz/datasets/; MOA SourceForge archive: https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/ | `elecNormNew.arff.zip` | Parse ARFF rows, convert nominal values to numeric codes, and keep the class label last. |
| `occupancy` | `dataset/occupancy.npy` | UCI Occupancy Detection: https://archive.ics.uci.edu/dataset/357/occupancy+detection | `datatraining.txt`, `datatest.txt`, `datatest2.txt` | Drop non-numeric ID/date columns unless your experiment explicitly uses encoded time features. |
| `shuttle` | `dataset/shuttle.npy` | UCI Statlog Shuttle: https://archive.ics.uci.edu/dataset/148/statlog+shuttle | `shuttle.trn.Z`, `shuttle.tst` | Combine train/test files if your experiment requires one continuous stream. |
| `kddcup99` | `dataset/kddcup99.npy` | KDD Cup 1999 archive: https://kdd.org/cupfiles/KDDCupData/1999/ | `kddcup.data_10_percent.zip` or `kddcup.data.zip` | Encode categorical columns such as protocol, service, flag, and attack labels before saving. |

## Reproducibility Notes

- Do not mix raw and normalized variants without recording the choice.
- Keep preprocessing scripts or notes with your experiment records.
- If you publish processed data separately, cite the original dataset provider.
- This repository intentionally keeps dataset files outside Git.

Always check each dataset's own license, citation, and redistribution terms
before publishing raw or processed copies. For this repository, keep benchmark
data outside Git and document the exact preprocessing steps used in experiment
records.
