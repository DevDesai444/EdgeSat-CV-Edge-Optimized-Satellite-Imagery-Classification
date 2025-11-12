# Data and environment preparation

## Repository and environment setup
Clone the repository using the remote URL you plan to work from, then switch
into the repository directory and choose whatever branching model fits your
workflow.

To set up the python environment, you can run:
```
make requirements
```
which will use the `env.yaml` file with `conda` to install the necessary
packages and to activate the right environment.

## Bucket mount and config setup
To mount your data, if you are on Google Cloud, you can simply mount a bucket
through the command:
```
gcsfuse --implicit-dirs my-bucket /path/to/mount/point
```

Once the data is mounted, some care has to be put to let the code know where
the data and folders are.
In particular the `config/config.yaml` file is used to link the right folders
for saving results and cached datamodules.

### Data folders
Particular care is to be taken for the data, depending ont he dataset of use.
In any case, the datamodule `ParsedDataModule` expect the events of interest
to be placed in a single folder, which will be its `root_folder`.
Specific datasets need a specific folder structure.
