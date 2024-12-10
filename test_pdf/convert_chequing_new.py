#!/usr/bin/env python
# coding: utf-8

# In[36]:


import datetime
import itertools
import json
import logging
import re
import sys
import typing
import xml.etree.ElementTree as ET

from decimal import Decimal
from functools import partial
from io import StringIO
from itertools import groupby, starmap, takewhile
from pathlib import Path
from typing import List, Iterator, NamedTuple

from dateutil.parser import parse as date_parse
from parsel import Selector
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams


# In[3]:


log = logging.getLogger("parser")
logging.basicConfig(stream=sys.stdout, level=logging.WARN)


# In[4]:


# Generate XML file:
# pdf2txt.py --output_type xml --outfile - -A -L 0.51 -F +0.8 -V test_may_2023.pdf | xmllint --format - > test_may_2023_new.xml


# In[5]:


class BBox(NamedTuple):
    left: float
    bottom: float
    right: float
    top: float

def process_bbox(tag: Selector) -> BBox:
    # Expect a comma-separated list of 4 coordinates in order: left, top, right, bottom
    # Parse them into a list of strings
    str_coords = tag.attrib.get("bbox", "0,0,0,0").split(",")
    # Use map to convert to a list of floats
    coords = map(float, str_coords)
    # Unpack coords
    return BBox(*coords)

def bounding_bbox(first: BBox, last: BBox) -> BBox:
    return BBox(first.left, first.bottom, last.right, last.top)


# In[6]:


class TextLine(NamedTuple):
    bbox: BBox
    parent_id: int
    text: str
    font: str
    size: str

def process_textline(textline: Selector) -> TextLine:
    bbox = process_bbox(textline)
    text = "".join(textline.xpath("./text/text()").getall()).strip()
    parent_id = int(textline.xpath("parent::textbox/@id").get())
    font = textline.xpath("text/@font").get()
    size = float(textline.xpath("text/@size").get())
    return TextLine(bbox, parent_id, text, font, size)


# In[7]:


class LinePart(NamedTuple):
    bbox: BBox

class Separator(NamedTuple):
    bbox: BBox

def process_linepart(linepart: Selector) -> LinePart:
    bbox = process_bbox(linepart)
    return LinePart(bbox)

def process_separator(lineparts: List[LinePart]) -> Separator:
    bbox = bounding_bbox(lineparts[0].bbox, lineparts[-1].bbox)
    return Separator(bbox)


# In[32]:


details_sentinel = "Details of your account"
date_matcher = re.compile(r"^From (.+) to (.+)$")
page_number_matcher = re.compile("^(\d+) of (\d+)$")

Component = TextLine|Separator
Components = List[Component]


class Page(NamedTuple):
    start_date: datetime.date
    end_date: datetime.date
    components: Components


def process_page_components(page_tag: Selector) -> Page:
    page_id = page_tag.attrib.get("id", "0")
    log.info(f"Processing page {page_id}")

    start_date = None
    end_date = None
    page_transaction_components = []
    sentinel_found = None
    
    # Find all textboxes that are not vertical and are larger than size 8
    for textline_tag in page_tag.xpath(".//textbox[not(@wmode = 'vertical')]/textline[text/@size >= 8]"):
        textline = process_textline(textline_tag)
        log.info(f"Textline: {textline.bbox}; {textline.font}@{textline.size}\n{textline.text}")
        page_transaction_components.append(textline)

        if textline.text.startswith(details_sentinel):
            log.info(f"sentinel found: {textline.text!r}")
            sentinel_found = textline
        if date_matcher.match(textline.text):
            log.info(f"found page dates: {textline}")
            try:
                start_date_str, end_date_str = date_matcher.match(textline.text).groups()
                start_date = date_parse(start_date_str).date()
                end_date = date_parse(end_date_str).date()
            except Exception as exc:
                log.warning(f"Could not parse dates: {exc}")
        elif page_number_matcher.match(textline.text):
            # Skip page numbers
            page_transaction_components.pop()

    sep_lines = []
    # Filter for all lines
    for line_tag in page_tag.xpath(".//line"):
        linepart = process_linepart(line_tag)
        
        # Ignore lines that are not below the sentinel
        # Vertical coordinates start from the bottom of the page
        if sentinel_found is not None and linepart.bbox.bottom > sentinel_found.bbox.bottom:
            continue

        first_linepart = next(iter(sep_lines), None)
        if first_linepart is None or first_linepart.bbox.bottom == linepart.bbox.bottom:
            sep_lines.append(linepart)
        else:
            sep = process_separator(sep_lines)
            log.info(f"Separator: {sep}")
            page_transaction_components.append(sep)
            sep_lines = [linepart]
    else:
        if sep_lines:
            sep = process_separator(sep_lines)
            page_transaction_components.append(sep)

    # Sort the parsed transaction components vertically (inversed), then horizontally
    page_transaction_components.sort(key=lambda item: (-(item.bbox.top), item.bbox.left))

    # Didn't find sentinel, so there probably aren't transactions here
    if sentinel_found is None:
        return Page(start_date=start_date, end_date=end_date, components=[])

    # Take all items after sentinel
    sentinel_index = page_transaction_components.index(sentinel_found)
    if sentinel_index == -1:
        raise
    if len(page_transaction_components) < sentinel_index+1:
        raise

    return Page(start_date=start_date, end_date=end_date, components=page_transaction_components[sentinel_index+1:])


# In[40]:


clean_date = re.compile(r"[$,]")

transaction_fields = ["date", "description", "withdrawals", "deposits", "balance"]
transaction_field_types = [datetime.date, str, Decimal, Decimal, Decimal]

class Transaction(NamedTuple):
    date: datetime.date
    description: str = ""
    withdrawals: Decimal = None
    deposits: Decimal = None
    balance: Decimal = None

def take_transaction_components(components: Iterator[Component]) -> Components:
    """
    Accumulate all the components considered part of a single transaction.
    A transaction is defined as all the Components until a Separator is encountered.
    """
    not_separator = lambda item: type(item) != Separator
    tx_components = []
    for item in takewhile(not_separator, components):
        tx_components.append(item)
    return tx_components

def validate_transaction_headers(header_text: List[TextLine]) -> bool:
    if len(header_text) != len(transaction_fields):
        return False

    # Ensure headers are Text types
    types_match = [type(text) == TextLine for text in header_text]
    if not all(types_match):
        return False
    
    # Ensure all header text fields match expected transaction fields
    # TODO: sort headers and fields?
    fields_match = [
        header_text.text.lower().startswith(fieldname)
        for header_text, fieldname in zip(header_text, transaction_fields)
    ]
    return all(fields_match)

def map_component_to_field(header_fields: dict[str, TextLine], component: TextLine) -> (str, TextLine):
    field_name = ""

    def is_aligned(field: str, a: Component, b: Component, threshold: float = 10.0):
        a_field = getattr(a.bbox, field)
        b_field = getattr(b.bbox, field)
        return abs(a_field - b_field) < threshold

    for field, field_type in zip(transaction_fields, transaction_field_types):
        header_component = header_fields.get(field, None)
        if not header_component:
            continue

        # Numeric fields will be right-aligned, so check if right bounds are close
        if field_type == Decimal and is_aligned("right", header_component, component, 2.0):
            field_name = field
            break
        elif is_aligned("left", header_component, component, 8.0):
            field_name = field
            break

    return field_name, component

def page_to_transactions(page: Page, description_separator=" ") -> List[Transaction]:
    it_components = iter(page.components)

    # First line should be headers
    header_components = take_transaction_components(it_components)

    if not validate_transaction_headers(header_components):
        raise Exception(f"Invalid headers! {header_components}")

    # Map fields to header components: {"date": TextLine, "description": TextLine, ...}
    header_fields = dict(zip(transaction_fields, header_components))
    header_mapper = partial(map_component_to_field, header_fields)
    transactions = []

    last_date = page.start_date
    tx_field_types = typing.get_type_hints(Transaction)
    while True:
        tx_components = take_transaction_components(it_components)
        if not tx_components:
            break

        tx_dict = {}
        # Collect final transaction component info
        # Final type conversion for dates and Decimal
        for component in tx_components:
            key, comp = header_mapper(component)
            key_type = tx_field_types.get(key, None)
            
            if key_type == str:
                tx_dict[key] = tx_dict.get(key, []) + [comp.text]
            if key_type == datetime.date:
                try:
                    date = date_parse(comp.text, default=last_date)

                    # This probably means we're in a new year
                    if date < last_date:
                        date = date.replace(year=page.end_date.year)

                    last_date = date
                    tx_dict[key] = date

                except Exception as exc:
                    log.warning(f"Bad date: {comp.text!r} ({exc})")
            if key_type == Decimal:
                try:
                    n = Decimal(clean_date.sub("", comp.text))
                    tx_dict[key] = n
                except:
                    log.warning(f"Bad decimal: {comp.text!r}")

        if tx_dict:
            if not "date" in tx_dict:
                tx_dict["date"] = last_date
            tx_dict["description"] = description_separator.join(tx_dict["description"])
            transactions.append(Transaction(**tx_dict))

    return transactions


# In[34]:


from tabulate import tabulate

def parse_xml_doc(filename: str|Path) -> List[Transaction]:
    with open(filename, "r") as fp:
        xmlraw = fp.read()

    try:
        doc = Selector(text=xmlraw, type="xml")
    except:
        log.error(f"Could not parse {filename} as XML")
        sys.exit(1)

    return parse_xml(doc)

def parse_xml(doc: Selector) -> List[Transaction]:
    all_transactions = []

    for page in doc.xpath("./page"):
        page_id = page.attrib.get("id", "0")
        log.info(f"Processing page {page_id}\n")
        page_comp = process_page_components(page)

        # Skip pages with no components (might just be text)
        if not page_comp.components:
            continue

        page_transactions = page_to_transactions(page_comp)
        all_transactions += page_transactions
    
    return all_transactions

def tabulate_results(res: List[Transaction]):
    # tabulate_tx = list(map(lambda tx: [tx.get(k, None) for k in transaction_fields], all_transactions))
    full_table = tabulate(
        res,
        headers=transaction_fields,
        tablefmt="rounded_grid",
        numalign="right",
        floatfmt=".2f",
    )

    print(full_table)

tabulate_results(parse_xml_doc(Path(".") / "test_dec_2022_new.xml"))


# In[37]:


def parse_pdf_doc(filename: str|Path) -> List[Transaction]:
    xml_data = StringIO()
    layout_params = {'detect_vertical': True, 'line_overlap': 0.51, 'all_texts': True, 'boxes_flow': 0.8}
    with open(filename, "rb") as fp:
        extract_text_to_fp(inf=fp, outfp=xml_data, output_type="xml", laparams=LAParams(**layout_params), codec=None)

    try:
        doc = Selector(text=xml_data.getvalue(), type="xml")
    except:
        log.error(f"Could not parse {filename} as XML")
        sys.exit(1)

    return parse_xml(doc)


# In[41]:


tabulate_results(parse_pdf_doc(Path(".") / "test_dec_2022.pdf"))


# In[ ]:




