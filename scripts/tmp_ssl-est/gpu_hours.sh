#!/bin/bash
# GPU-hours of this goal's jobs since 03:12 (all attempts, preempted ones included), by job name and partition.
sacct -u $USER -S 2026-11-06T03:12 --duplicates -X -P -n --format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES \
  | awk -F'|' '$2 ~ /^(cf-data|ssl-est|shape-rl|cf-data-)/ {g=0; if (match($6, /gres\/gpu=[0-9]+/)) g=substr($6, RSTART+9, RLENGTH-9);
      h[$2" "$3]+=g*$5/3600; n[$2" "$3]++; t+=g*$5/3600} END {for (k in h) printf "%-28s attempts %3d  GPU-h %7.2f\n", k, n[k], h[k]; printf "TOTAL GPU-h %.2f\n", t}' | sort
