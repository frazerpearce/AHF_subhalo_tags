# AHF subhalo tagging

Utility for rehashing AHF halo catalogues, appending revised subhalo ids and counts.

## Files
- `fix_ahf_subhalo_tags.py`

## Usage
```bash
python fix_ahf_subhalo_tags.py input.AHF_halos

Rehash an AHF halos catalogue to append direct host/subhalo information.

For each halo:
- main halo: its centre does not lie within the search radius of any larger halo
             (where "larger" follows the user's convention: smaller haloid)
- subhalo:   its centre lies within the search radius of one or more larger halos

Two host-assignment schemes are written:

1) Standard centre-in-r200 criterion:
       dist <= r200_host
   producing columns:
       sub_count  sub_id

2) AHF subhalo link criterion:
       dist <= sqrt(r200_host^2 + 0.5 * r200_sub^2)
   producing columns:
       sub_count_wide  sub_id_wide

Among all larger halos that contain a halo centre, the assigned host is the
*smallest* such host halo, i.e. the containing halo with the largest haloid.
Main halos get host id 0.

The script preserves the original file line-by-line and appends four extra columns:
    sub_count  sub_id  sub_count_wide  sub_id_wide

Assumptions:
- periodic cube with boxsize = 1e6 kpc by default
- AHF halo columns are, by default:
    haloid = col 0
    x,y,z  = cols 5,6,7
    r200   = col 11
