# data/

The challenge data is **not** included in this repository (it belongs to the challenge organizer, Raidium, and is distributed through the Challenge Data platform).

To run the notebooks, download the files from the challenge page and place them here:

```
data/
├── train-images/          # training CT slices, one PNG per image (0.png, 1.png, ...)
├── test-images/           # test CT slices (only used by notebooks/early_tests/)
├── y_train.csv            # training labels (one row per image, 65,536 pixel columns)
└── annotated_labels.json  # already here: for each image, the list of organ IDs that were annotated
```

`annotated_labels.json` is a file provided by the challenge (see `docs/challenge_description_fr.txt`). It is a list of 2,000 entries; entry *i* lists the structure IDs that are annotated in training image *i*.
