#!/bin/bash
# Stage 10: raw array bandwidth with fio (direct I/O): sequential read at 1M/512K/256K, and
# read while a rate-limited writer runs. Skipped if fio is missing.
source "$(dirname "$0")/lib.sh"
stage_begin 10_fio
command -v fio >/dev/null || { stage_end 10_fio skip "fio not installed"; exit 0; }
F="$DATA_DIR/fio"; mkdir -p "$F"
COMMON="--filename_format=f.\$jobnum --directory=$F --direct=1 --ioengine=libaio --group_reporting --output-format=json"
# 8 jobs x 4 GiB files; laid out once
fio --name=layout --rw=write --bs=1M --size=4G --numjobs=8 $COMMON --output="$SDIR/layout.json" >/dev/null 2>&1
for bs in 1M 512K 256K; do
  for rep in 1 2; do
    fio --name=read_$bs --rw=read --bs=$bs --iodepth=32 --numjobs=8 --size=4G \
        --time_based --runtime=8 $COMMON --output="$SDIR/read_${bs}_r$rep.json" ||
      { stage_end 10_fio fail "fio read $bs failed"; exit 1; }
  done
done
read_mbs=$(python3 - "$SDIR" <<'EOF'
import json, glob, statistics, sys
v = [json.load(open(p))["jobs"][0]["read"]["bw_bytes"] / 1e6 for p in glob.glob(sys.argv[1] + "/read_1M_r*.json")]
print(int(statistics.median(v)))
EOF
)
echo "$read_mbs" > "$SDIR/read_1M_median_MBs.txt"
WB=${WB_RATE_MBS:-$((read_mbs / 5))}
echo "$WB" > "$SDIR/writeback_rate_MBs.txt"
# read under a concurrent writer limited to WB MB/s (written into separate files)
fio --name=rd --filename_format='f.$jobnum' --rw=read --bs=1M --iodepth=32 --numjobs=8 --size=4G --time_based --runtime=12 \
    --directory=$F --direct=1 --ioengine=libaio --group_reporting \
    --name=wr --new_group --rw=write --bs=1M --iodepth=8 --numjobs=2 --size=4G --time_based --runtime=12 \
    --rate=$((WB / 2))m --filename_format='wr.$jobnum' --directory=$F --direct=1 --ioengine=libaio \
    --output-format=json --output="$SDIR/read_under_write.json" ||
  { stage_end 10_fio fail "fio mixed failed"; exit 1; }
rm -rf "$F"
stage_end 10_fio ok "seq read 1M median ${read_mbs} MB/s; writeback rate ${WB} MB/s"
