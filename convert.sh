#!/bin/bash -e
#

PATH="./venv/bin/:$PATH"
export PYTHONPATH="./venv/lib/python3.10/site-packages/"

# define output filename
OUTPUT_FILE='transactions.csv'

# find pdf2txt.py
PDF2TXT=$(which pdf2txt.py)

PDF_FILES="$(find -H . -type f -regextype posix-extended -regex '\./.+\.pdf' -printf '%f|\n')"
echo "$PDF_FILES"

while IFS='|' read -r PDF;
do
	XML_FILE=$(basename -s .pdf "${PDF}")
	./venv/bin/python "${PDF2TXT}" -o "${XML_FILE}.xml" "${PDF}" 
done <<< "$PDF_FILES"

readarray -d '' XML_FILES < <(find -H . -maxdepth 1 -type f -regextype posix-extended -regex '\./.+\.xml' -print0)

./venv/bin/python convert.py "${OUTPUT_FILE}" "${XML_FILES[@]}"

rm -rf *.xml
