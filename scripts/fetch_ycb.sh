#!/bin/bash
# Download another YCB object (google_16k scan) into assets/ycb/<id>/ and build its physics mesh.
# Planning a grasp for it then needs the 'plan' extra (open3d):
#   python -m revo3_grasp.planning.plan_grasp --mesh assets/ycb/<id>/physics.ply --output assets/ycb/<id>/plan.json --orientations all
set -euo pipefail
id=$1   # e.g. 014_lemon
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
curl -sSfL -o "$tmp/$id.tgz" "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/${id}_google_16k.tgz"
tar -xzf "$tmp/$id.tgz" -C "$tmp"
mkdir -p "$root/assets/ycb/$id"
cp "$tmp/$id/google_16k/"{textured.obj,textured.mtl,texture_map.png} "$root/assets/ycb/$id/"
python -m revo3_grasp.planning.prep_object "$tmp/$id/google_16k" "$root/assets/ycb/$id/physics.ply"
rm -rf "$tmp"
