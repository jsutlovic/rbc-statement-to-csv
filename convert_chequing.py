#!/usr/bin/env python3
from datetime import datetime
from dateutil.parser import parse
import sys
import xml.etree.ElementTree as ET
import re
import csv

output_file = sys.argv[1]
input_files = sys.argv[2:]

font_header = "MetaBookLF-Roman"
font_txn = "MetaBoldLF-Roman"

class Block:
    def __init__(self, page, x, x2, y, text):
        self.page = page
        self.x = x
        self.x2 = x2
        self.text = text
        self.y = y
    
    def __repr__(self):
        # return self.text
        return f"<Block page={self.page} x={self.x} x2={self.x2} y={self.y} text={self.text} />"

csv_rows = []

for input_file in input_files:
    tree = ET.parse(input_file)
    root = tree.getroot()
    rows = []

    continue_input_loop = False
    for i_tag, tag in enumerate(root[0][1]):
        if i_tag > 10:
            continue_input_loop = True
            break
        if font := tag.get("font"):
            if font.endswith("MetaBoldLF-Roman") or font.endswith("Utopia-Bold"):
                break
    
    if continue_input_loop:
        print(f"Skipping {input_file}...")
        continue

    print(f'Processing {input_file}...')

    blocks = []
    pages = set()
    # Go through each page
    for page_num, page in enumerate(root):
        pages.add(page_num)
        # Txn rows are in the second figure
        figure = page[1]
        text = ''
        last_x = None
        last_x2 = None
        block_x = None
        width = 0
        seen_text = False
        # A row is a list of <text> tags, each containing a character
        for tag in figure:
            clear_text = False
            append_block = ""
            if tag.tag == 'text':
                seen_text = True
                # Filter on text size to remove some of the noise
                font = tag.attrib.get("font")
                if font:
                    font = font.split("+")[1]
                size = float(tag.attrib['size'])
                x_pos = float(tag.attrib["bbox"].split(",")[0])
                y_pos = float(tag.attrib["bbox"].split(",")[1])
                x2_pos = float(tag.attrib["bbox"].split(",")[2])

                if last_x2 is not None:
                    # if x2_pos < last_x2:
                    #     clear_text = True

                    if x_pos - last_x2 > 5:
                        append_block = text
                        text = ""
                        
                        width = 0
                    elif (x_pos - last_x2) > 0.7:
                        text += " "
                last_x = x_pos
                last_x2 = x2_pos
                width += size
                if font in (font_txn, font_header):
                    text += tag.text
                if block_x is None:
                    block_x = x_pos
            elif tag.tag != 'text' and text != '':
                # Row is over, start a new one
                if seen_text:
                    append_block = text
                seen_text = False
                clear_text = True
                width = 0
                last_x2 = None
            
            if append_block:
                block = Block(page_num, block_x, x2_pos, y_pos, append_block.strip())
                blocks.append(block)
                block_x = None

            if clear_text:
                text = ''
            
    open_balance_parts = [b.text for b in blocks if b.text.startswith("Your opening balance")][0].split(" ")[-3:]
    open_balance_date = parse(" ".join(open_balance_parts))
    start_year = int(open_balance_parts[2])
    print(open_balance_date, start_year)

    header_sets = []
    for page in pages:
        page_blocks = [b for b in blocks if b.page == page]
        end_of_header_index = 0

        for i, block in enumerate(page_blocks):
            if block.text == "Date":
                if other_blocks := page_blocks[i+1:i+5]:
                    header_sets.append([block, *other_blocks])
                    end_of_header_index = i + 4

        page_blocks = [block for i, block in enumerate(page_blocks) if i > end_of_header_index]

        print(header_sets)
        if len(header_sets) <= page:
            break

        cell_pos = 0
        i = 0
        block_pos = 0
        row = []
        last_date = None

        # i counts each cell, working ltr and down
        # block_pos counts the blocks we have scraped from the xml file
        # if the position of the block does not match the cell we are in,
        # we iterate i but not block pos, and print a blank cell (or date)
        while block_pos < len(page_blocks):
            row_pos = 0
            block = page_blocks[block_pos]
            block_consumed = False

            headers = header_sets[block.page]

            # use the block to determine what column its in
            mid_point = (block.x2 - block.x) / 2 + block.x
            for header in headers:
                if mid_point > header.x and mid_point < header.x2:
                    row_pos = headers.index(header)

            # check the header identified by block matches the cell we're looking at
            if i % 5 == row_pos:
                if i % 5 == 0:
                    date = parse(f"{block.text} {start_year}")

                    # handle year rollover
                    if date < open_balance_date:
                        date = parse(f"{block.text} {start_year+1}")
                    block.text = str(date.date())
                    if block.text.strip():
                        last_date = block.text
                block_consumed = True
                row.append(block.text)
            elif i % 5 == 0 and page_blocks[block_pos].text == "Opening Balance":
                row.append(str(open_balance_date.date()))
            # if its the first cell, and we have a date from the last row, use that
            elif last_date and i % 5 == 0:
                row.append(last_date)
            # if we didn't have a block for this cell, we just print an empty cell
            else:
                row.append("")
            if i % 5 == 4:
                csv_rows.append(row)
                row = []
            if block_consumed:
                block_pos += 1
            i += 1

# Write as csv
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([
        "Date",
        "Description",
        "Withdrawls",
        "Deposits",
        "Balance"
    ])

    for row in csv_rows:
        csv_writer.writerow(row)