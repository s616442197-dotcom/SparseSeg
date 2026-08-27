#!/usr/bin/env bash
# Historical filename retained for compatibility. The implementation now uses
# only the pip-installed official nnU-Net v2 pipeline in nnunetv2_official/.

set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$here/nnunetv2_official/submit_official_nnunetv2_pipeline.sh" "$@"
