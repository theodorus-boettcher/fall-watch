#!/usr/bin/env bash
# Download ZeroWaste (real recycling-plant belt footage) into data/zerowaste and check the md5 sums.
# -f: 4,503 frames labeled by humans (train/val/test). -s: 6,212 unlabeled frames (split zip archive).
# Licence: Zenodo lists CC BY 4.0, the project page (ai.bu.edu/zerowaste) says CC BY-NC 4.0; we follow
# the stricter one. Safe to run again: finished files are skipped, broken downloads resume.
# Then it unpacks -f into splits_final_deblurred/ and -s into zerowaste-s/. The -s archive is
# encrypted; the authors publish the password in the Zenodo record description.
set -euo pipefail
dest="$(cd "$(dirname "$0")/.." && pwd)/data/zerowaste"
base=https://zenodo.org/api/records/6412647/files
mkdir -p "${dest}"

fetch() {  # fetch <file> <md5>
  local file="$1" md5="$2"
  if [ -f "${dest}/${file}" ] && echo "${md5}  ${dest}/${file}" | md5sum -c --status; then
    echo "ok (already there) ${file}"
    return 0
  fi
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "${dest}/${file}" "${base}/${file}/content"
  echo "${md5}  ${dest}/${file}" | md5sum -c
}

fetch zerowaste-f-final.zip e26e31a58080bca6782dca0e56074c5d
fetch zerowaste-s-parts.zip 10a38443fe43eedad4fecadbcbe86589
fetch zerowaste-s-parts.z01 7d39e260b7f6b4ee7a9b078c923176cc
fetch zerowaste-s-parts.z02 47065ff56e2eb895c78587636f0408d2
fetch zerowaste-s-parts.z03 d7287fd9dcbf9ade985e5d8fe3850800
echo "all ZeroWaste files downloaded to ${dest}"

password='<password from the Zenodo record>'  # from the Zenodo record description
if [ ! -d "${dest}/splits_final_deblurred" ]; then
  unzip -q -P "${password}" "${dest}/zerowaste-f-final.zip" -d "${dest}"
fi
if [ ! -d "${dest}/zerowaste-s" ]; then
  zip -q -s 0 "${dest}/zerowaste-s-parts.zip" --out "${dest}/zerowaste-s-joined.zip"
  mkdir "${dest}/zerowaste-s.partial"
  unzip -q -P "${password}" "${dest}/zerowaste-s-joined.zip" -d "${dest}/zerowaste-s.partial"
  rm "${dest}/zerowaste-s-joined.zip"
  mv "${dest}/zerowaste-s.partial" "${dest}/zerowaste-s"
fi
echo "unpacked: $(find "${dest}/splits_final_deblurred" "${dest}/zerowaste-s" -type f | wc -l) files"
