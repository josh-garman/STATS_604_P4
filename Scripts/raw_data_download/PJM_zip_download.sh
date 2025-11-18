#!/usr/bin/env bash

# prerequisites
for cmd in curl unzip tar; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[error] missing '$cmd'"; exit 127; }
done

set -euo pipefail

#URL / paths
URL="https://files.osf.io/v1/resources/Py3u6/providers/osfstorage/?zip="
ZIP_OUT="Data/raw/osf_bundle.zip"
EXTRACT_DIR="Data/raw"
# ---------------------------------

mkdir -p "$(dirname "$ZIP_OUT")" "$EXTRACT_DIR"

tmp="${ZIP_OUT}.part"
#echo "[1/3] Downloading..."
# curl -L -C - -o "$tmp" "$URL"
curl -sS -L -C - -o "$tmp" "$URL"
mv -f "$tmp" "$ZIP_OUT"

#echo "[2/3] Verifying zip..."
unzip -tqq "$ZIP_OUT"

#echo "[3/3] Extracting to ${EXTRACT_DIR}..."
unzip -q -o "$ZIP_OUT" -d "$EXTRACT_DIR"

# OPTIONAL: if the zip contains a .tar.gz inside and you only want the CSVs, expand those too
if find "$EXTRACT_DIR" -type f -iname '*.tar.gz' -print -quit | grep -q .; then
  #echo "[info] Found nested .tar.gz — extracting..."
  while IFS= read -r -d '' tgz; do
    tar -xzf "$tgz" -C "$EXTRACT_DIR"
    rm -f "$tgz"
  done < <(find "$EXTRACT_DIR" -type f -iname '*.tar.gz' -print0)
fi


# optional: delete zip after extract
# rm -f "$ZIP_OUT"

#echo "[ok] Done. Files are in: ${EXTRACT_DIR}"