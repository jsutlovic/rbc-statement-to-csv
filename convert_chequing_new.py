from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from functools import partial
from io import StringIO
from itertools import takewhile
import logging
from pathlib import Path
import re
import sys
import typing
from typing import (
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
)

import click
from dateutil.parser import parse as date_parse
from parsel import Selector
from tabulate import tabulate
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams


# Generate XML file:
# pdf2txt.py --output_type xml --outfile - -A -L 0.51 -F +0.8 -V test_may_2023.pdf | xmllint --format - > test_may_2023_new.xml


log = logging.getLogger("convert_chequing")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

textline_xpath = ".//textbox[not(@wmode = 'vertical')]/textline[text/@size >= 8]"
details_sentinel = "Details of your account"
date_matcher = re.compile(r"^From (.+) to (.+)$")
page_number_matcher = re.compile(r"^(\d+) of (\d+)$")


class ParseException(Exception):
    pass


def check_type(t: type, tt: type) -> bool:
    if get_origin(tt) is Union:
        if t in get_args(tt):
            return True
    if t == tt:
        return True

    return False


class BBox(NamedTuple):
    left: float
    bottom: float
    right: float
    top: float

    @staticmethod
    def from_tag(tag: Selector) -> BBox:
        # Expect a comma-separated list of 4 coordinates in order: left, top, right, bottom
        # Parse them into a list of strings
        str_coords = tag.attrib.get("bbox", "0,0,0,0").split(",")
        # Use map to convert to a list of floats
        coords = map(float, str_coords)
        # Unpack coords
        return BBox(*coords)

    @staticmethod
    def get_bounding(first: BBox, last: BBox) -> BBox:
        return BBox(first.left, first.bottom, last.right, last.top)


class TextLine(NamedTuple):
    bbox: BBox
    parent_id: int
    text: str
    font: str
    size: float

    @staticmethod
    def from_tag(tag: Selector) -> TextLine:
        bbox = BBox.from_tag(tag)
        text = "".join(tag.xpath("./text/text()").getall()).strip()
        parent_id = int(tag.xpath("parent::textbox/@id").extract_first(default="0"))
        font = tag.xpath("text/@font").extract_first(default="Unknown")
        size = float(tag.xpath("text/@size").extract_first(default="0"))
        return TextLine(bbox, parent_id, text, font, size)


class LinePart(NamedTuple):
    bbox: BBox

    @staticmethod
    def from_tag(linepart: Selector) -> LinePart:
        bbox = BBox.from_tag(linepart)
        return LinePart(bbox)


class Separator(NamedTuple):
    bbox: BBox

    @staticmethod
    def from_lineparts(lineparts: List[LinePart]) -> Separator:
        bbox = BBox.get_bounding(lineparts[0].bbox, lineparts[-1].bbox)
        return Separator(bbox)


Component = Union[TextLine, Separator]
Components = Sequence[Component]


class Page(NamedTuple):
    start_date: date
    end_date: date
    components: Components


def process_page_components(page_tag: Selector) -> Page:
    page_id = page_tag.attrib.get("id", "0")
    log.info(f"Processing page {page_id}")

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    page_transaction_components = []
    sentinel_found = None

    # Find all textboxes that are not vertical and are larger than size 8
    for textline_tag in page_tag.xpath(textline_xpath):
        textline = TextLine.from_tag(textline_tag)
        log.info(
            f"Textline: {textline.bbox}; {textline.font}@{textline.size}\n{textline.text}"
        )
        page_transaction_components.append(textline)

        if textline.text.startswith(details_sentinel):
            log.info(f"sentinel found: {textline.text!r}")
            sentinel_found = textline

        date_matches = date_matcher.match(textline.text)
        if date_matches:
            log.info(f"found page dates: {textline}")
            try:
                start_date_str, end_date_str = date_matches.groups()
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
        linepart = LinePart.from_tag(line_tag)

        # Ignore lines that are not below the sentinel
        # Vertical coordinates start from the bottom of the page
        if (
            sentinel_found is not None
            and linepart.bbox.bottom > sentinel_found.bbox.bottom
        ):
            continue

        first_linepart = next(iter(sep_lines), None)
        if first_linepart is None or first_linepart.bbox.bottom == linepart.bbox.bottom:
            sep_lines.append(linepart)
        else:
            sep = Separator.from_lineparts(sep_lines)
            log.info(f"Separator: {sep}")
            page_transaction_components.append(sep)
            sep_lines = [linepart]
    else:
        if sep_lines:
            sep = Separator.from_lineparts(sep_lines)
            page_transaction_components.append(sep)

    # Raise if we can't find transaction dates
    if type(start_date) is None or type(end_date) is None:
        raise ParseException("Unable to find page date")

    # Explicit typing fix for the above check
    start_date = cast(date, start_date)
    end_date = cast(date, end_date)

    # Sort the parsed transaction components vertically (inversed), then horizontally
    page_transaction_components.sort(
        key=lambda item: (-(item.bbox.top), item.bbox.left)
    )

    # Didn't find sentinel, so there probably aren't transactions here
    if sentinel_found is None:
        return Page(start_date=start_date, end_date=end_date, components=[])

    # Take all items after sentinel
    sentinel_index = page_transaction_components.index(sentinel_found)
    if sentinel_index == -1:
        raise
    if len(page_transaction_components) < sentinel_index + 1:
        raise

    return Page(
        start_date=start_date,
        end_date=end_date,
        components=page_transaction_components[sentinel_index + 1 :],
    )


decimal_replace = re.compile(r"[$,]")

# TODO: Remove these and use Transaction typing hints
transaction_fields = ["date", "description", "withdrawals", "deposits", "balance"]
transaction_field_types = [datetime, str, Decimal, Decimal, Decimal]


class Transaction(NamedTuple):
    date: date
    description: str = ""
    withdrawals: Optional[Decimal] = None
    deposits: Optional[Decimal] = None
    balance: Optional[Decimal] = None


def take_transaction_components(components: Iterator[Component]) -> Sequence[TextLine]:
    """
    Accumulate all the components considered part of a single transaction.
    A transaction is defined as all the Components until a Separator is encountered.
    """
    not_separator = lambda item: type(item) != Separator
    tx_components = []
    for item in takewhile(not_separator, components):
        tx_components.append(item)
    return tx_components


def validate_transaction_headers(header_text: Components) -> bool:
    if len(header_text) != len(transaction_fields):
        return False

    # Ensure headers are Text types
    types_match = [type(text) is TextLine for text in header_text]
    if not all(types_match):
        return False

    header_text = cast(List[TextLine], header_text)

    # Ensure all header text fields match expected transaction fields
    fields_match = [
        header_text.text.lower().startswith(fieldname)
        for header_text, fieldname in zip(header_text, transaction_fields)
    ]
    return all(fields_match)


def map_component_to_field(
    header_fields: Dict[str, TextLine], component: TextLine
) -> Tuple[str, TextLine]:
    field_name = ""

    log.info(f"Mapping component to header: {component}")

    def is_aligned(field: str, a: Component, b: Component, threshold: float = 10.0):
        a_field = getattr(a.bbox, field)
        b_field = getattr(b.bbox, field)
        return abs(a_field - b_field) < threshold

    for field, field_type in zip(transaction_fields, transaction_field_types):
        header_component = header_fields.get(field, None)
        if not header_component:
            continue

        # Numeric fields will be right-aligned, so check if right bounds are close
        if issubclass(Decimal, field_type) and is_aligned(
            "right", header_component, component, 2.0
        ):
            field_name = field
            break
        elif is_aligned("left", header_component, component, 8.0):
            field_name = field
            break

    log.info(f"Mapped to {field_name}")

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

    # Use datetime to avoid issues with date_parse()
    last_date = datetime.combine(page.start_date, datetime.min.time())
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
            key_type = tx_field_types.get(key, str)
            log.info(f"Got key {key} with key_type {key_type} for {comp.text!r}")

            if check_type(str, key_type):
                tx_dict[key] = tx_dict.get(key, []) + [comp.text]
            if check_type(date, key_type):
                try:
                    tx_date = date_parse(comp.text, default=last_date)

                    # This probably means we're in a new year
                    if tx_date < last_date:
                        tx_date = tx_date.replace(year=page.end_date.year)

                    last_date = tx_date
                    tx_dict[key] = tx_date.date()

                except Exception as exc:
                    log.warning(f"Bad date: {comp.text!r} ({exc})")
            if check_type(Decimal, key_type):
                try:
                    n = Decimal(decimal_replace.sub("", comp.text))
                    tx_dict[key] = n
                except:
                    log.warning(f"Bad decimal: {comp.text!r}")

        if tx_dict:
            log.info(f"Got transaction: {tx_dict}")
            if not "date" in tx_dict:
                tx_dict["date"] = last_date.date()
            tx_dict["description"] = description_separator.join(tx_dict["description"])
            transactions.append(Transaction(**tx_dict))

    return transactions


def parse_xml_doc(filename: str | Path) -> List[Transaction]:
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
    log.info(f"First transaction: {res[0]}")
    full_table = tabulate(
        res,
        headers=transaction_fields,
        tablefmt="rounded_grid",
        numalign="right",
        floatfmt=".2f",
    )

    print(full_table)


def parse_pdf_doc(filename: Path) -> List[Transaction]:
    xml_data = StringIO()
    layout_params = {
        "detect_vertical": True,
        "line_overlap": 0.51,
        "all_texts": True,
        "boxes_flow": 0.8,
    }
    with filename.open(mode="rb") as fp:
        extract_text_to_fp(
            inf=fp,
            outfp=xml_data,
            output_type="xml",
            laparams=LAParams(**layout_params),
            codec="",
        )

    try:
        doc = Selector(text=xml_data.getvalue(), type="xml")
    except:
        log.error(f"Could not parse {filename} as XML")
        sys.exit(1)

    return parse_xml(doc)


@click.command()
@click.argument(
    "pdf_file",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path),
)
def convert(pdf_file: Path):
    """Converts RBC Chequing account statements to machine-readable formats"""
    transactions = parse_pdf_doc(pdf_file)
    tabulate_results(transactions)


if __name__ == "__main__":
    convert()
