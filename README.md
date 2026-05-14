# MRMS Precipitation Nowcasting

This repository contains deep learning models for short-term MRMS precipitation nowcasting using 3-hour radar precipitation cubes.

## Models

This repository includes:

1. ConvLSTM baseline
2. DDPM nowcasting model
3. Physics-guided DDPM nowcasting model

## Data

The full MRMS dataset is not included in this repository due to size.

Place local MRMS `.npz` files under:

```text
data/mrms_3hr_cubes_128/
